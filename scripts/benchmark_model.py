from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.models import build_model  # noqa: E402


def gpu_process_snapshot() -> list[dict[str, object]] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    processes: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Unexpected nvidia-smi process row: {line}")
        processes.append(
            {
                "pid": int(parts[0]),
                "process_name": parts[1],
                "used_gpu_memory_mib": float(parts[2]),
            }
        )
    return processes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model inference and FLOPs")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--mode", choices=("single", "both"), default="both")
    return parser.parse_args()


def load_model(
    model_config: dict[str, object], checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    model = build_model(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def active_parameter_count(models: list[torch.nn.Module], forward: Callable[[], object]) -> int:
    """Count unique parameters in modules executed by one inference call."""
    active: set[int] = set()

    def mark(module: torch.nn.Module, _inputs: object, _output: object) -> None:
        active.update(id(parameter) for parameter in module.parameters(recurse=False))

    handles = [module.register_forward_hook(mark) for model in models for module in model.modules()]
    try:
        forward()
    finally:
        for handle in handles:
            handle.remove()
    return sum(
        parameter.numel()
        for model in models
        for parameter in model.parameters()
        if id(parameter) in active
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError("batch size and repeats must be positive; warmup must be non-negative")
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device(args.device)
    detection_model = load_model(config["model"], args.checkpoint, device)
    segmentation_model = (
        load_model(config["model"], args.segmentation_checkpoint, device)
        if args.segmentation_checkpoint is not None
        else detection_model
    )
    image = torch.randn(
        args.batch_size, 3, args.height, args.width, dtype=torch.float32, device=device
    )
    target_ids = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    detection_ids = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    segmentation_ids = torch.ones(args.batch_size, dtype=torch.long, device=device)
    is_specialist_pair = args.segmentation_checkpoint is not None
    run_both_tasks = args.mode == "both"

    def forward() -> None:
        detection_model(image, detection_ids, target_ids)
        if run_both_tasks:
            segmentation_model(image, segmentation_ids, target_ids)

    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    gpu_processes_before: list[dict[str, object]] | None = None
    gpu_processes_after: list[dict[str, object]] | None = None
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ),
    ):
        models = [detection_model]
        if is_specialist_pair:
            models.append(segmentation_model)
        active_parameters = active_parameter_count(models, forward)
        for _ in range(args.warmup):
            forward()
        if device.type == "cuda":
            gpu_processes_before = gpu_process_snapshot()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeats):
                forward()
            end.record()
            torch.cuda.synchronize(device)
            latency_ms = start.elapsed_time(end) / args.repeats
            peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2
            gpu_processes_after = gpu_process_snapshot()
        else:
            import time

            started = time.perf_counter()
            for _ in range(args.repeats):
                forward()
            latency_ms = (time.perf_counter() - started) * 1000.0 / args.repeats
            peak_memory_mib = None

        flop_count = None
        try:
            from torch.utils.flop_counter import FlopCounterMode

            with FlopCounterMode(display=False) as counter:
                forward()
            flop_count = int(counter.get_total_flops())
        except (ImportError, RuntimeError, TypeError):
            pass

    parameters = sum(parameter.numel() for model in models for parameter in model.parameters())
    competing_by_pid: dict[int, dict[str, object]] = {}
    for snapshot in (gpu_processes_before, gpu_processes_after):
        if snapshot is None:
            continue
        for process in snapshot:
            if int(process["pid"]) != os.getpid():
                competing_by_pid[int(process["pid"])] = process
    latency_uncontended = device.type != "cuda" or (
        gpu_processes_before is not None
        and gpu_processes_after is not None
        and not competing_by_pid
    )
    output = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp_dtype": "bfloat16" if use_amp else "float32",
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "specialist_pair": is_specialist_pair,
        "mode": args.mode,
        "parameters": parameters,
        "active_inference_parameters": active_parameters,
        "inactive_training_only_parameters": parameters - active_parameters,
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_image": latency_ms / args.batch_size,
        "peak_gpu_memory_mib": peak_memory_mib,
        "flops_per_batch": flop_count,
        "flops_per_image": flop_count / args.batch_size if flop_count is not None else None,
        "gpu_processes_before": gpu_processes_before,
        "gpu_processes_after": gpu_processes_after,
        "competing_gpu_processes": list(competing_by_pid.values()),
        "latency_uncontended": latency_uncontended,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
