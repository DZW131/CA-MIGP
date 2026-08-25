from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data import ManifestTaskDataset  # noqa: E402
from pathm.data.manifest import manifest_sha256  # noqa: E402
from pathm.models import build_model  # noqa: E402
from pathm.training import MixedTaskLoss, evaluate, train_one_epoch  # noqa: E402


PROVENANCE_SOURCE_PATHS = (
    "scripts/train.py",
    "src/pathm/data/dataset.py",
    "src/pathm/data/manifest.py",
    "src/pathm/models/__init__.py",
    "src/pathm/models/baselines.py",
    "src/pathm/models/geometry.py",
    "src/pathm/models/network.py",
    "src/pathm/models/prompts.py",
    "src/pathm/training/engine.py",
    "src/pathm/training/losses.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unified prompted model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_source_provenance(config_path: Path) -> dict[str, object]:
    source_sha256 = {
        relative_path: file_sha256(REPOSITORY_ROOT / relative_path)
        for relative_path in PROVENANCE_SOURCE_PATHS
    }
    return {
        "captured_at_unix": time.time(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "source_sha256": source_sha256,
    }


def random_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"].cpu())  # type: ignore[union-attr]
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [value.cpu() for value in state["cuda"]]  # type: ignore[union-attr]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(checkpoint: dict[str, object], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def make_loader(
    config: dict[str, object],
    split_key: str,
    augment: bool,
    seed: int,
    mismatched_style: bool = False,
) -> DataLoader:
    data = config["data"]
    model = config["model"]
    training = config["training"]
    assert isinstance(data, dict) and isinstance(model, dict) and isinstance(training, dict)
    max_key = f"max_{split_key}_records"
    dataset = ManifestTaskDataset(
        manifest_path=data["manifest"],
        splits=data[f"{split_key}_split"],
        augment=augment,
        heatmap_sigma=float(data.get("heatmap_sigma", 2.0)),
        color_jitter_strength=float(data.get("color_jitter_strength", 0.0)) if augment else 0.0,
        hed_stain_jitter_strength=(
            float(data.get("hed_stain_jitter_strength", 0.0)) if augment else 0.0
        ),
        max_records=data.get(max_key),
        operations=data.get("operations"),
        stain_h_scale=float(data.get("stain_h_scale", 1.0)),
        stain_e_scale=float(data.get("stain_e_scale", 1.0)),
        mismatched_style=mismatched_style,
        include_geometry_targets=(
            augment
            and (
                bool(model.get("use_geometry_prototypes", False))
                or float(training.get("uncertainty_boundary_alpha", 0.0)) > 0.0
            )
        ),
        geometry_target_mode=str(data.get("geometry_target_mode", "instance")),
        geometry_rim_threshold=float(data.get("geometry_rim_threshold", 0.15)),
        geometry_shell_threshold=float(data.get("geometry_shell_threshold", 0.35)),
        geometry_contact_radius=float(data.get("geometry_contact_radius", 3.0)),
        geometry_near_background_radius=float(
            data.get("geometry_near_background_radius", 3.0)
        ),
        geometry_gap_kernel_sizes=data.get("geometry_gap_kernel_sizes", (1, 2, 3)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=augment,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(data.get("num_workers", 0)) > 0,
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    launch_provenance = capture_source_provenance(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    training = config["training"]
    evaluation = config["evaluation"]
    assert isinstance(training, dict) and isinstance(evaluation, dict)
    seed = int(training["seed"])
    set_seed(seed, bool(training.get("deterministic", False)))

    device = torch.device(args.device)
    train_loader = make_loader(config, "train", augment=True, seed=seed)
    val_loader = make_loader(config, "val", augment=False, seed=seed)
    model = build_model(config["model"]).to(device)
    uncertainty_state_indices = training.get(
        "uncertainty_boundary_state_indices", (0, 1, 4)
    )
    if not isinstance(uncertainty_state_indices, (list, tuple)):
        raise ValueError("uncertainty_boundary_state_indices must be a list or tuple")
    training_loss_function = MixedTaskLoss(
        uncertainty_boundary_alpha=float(
            training.get("uncertainty_boundary_alpha", 0.0)
        ),
        uncertainty_boundary_mode=str(
            training.get("uncertainty_boundary_mode", "none")
        ),
        uncertainty_boundary_state_indices=tuple(
            int(index) for index in uncertainty_state_indices
        ),
    )
    validation_loss_function = MixedTaskLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"])
    )

    output_root = Path(config["output_dir"])
    run_name = str(config["run_name"])
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=args.resume is not None)
    resolved_config = yaml.safe_dump(config, sort_keys=False)
    stored_config_path = run_dir / "config.yaml"
    if args.resume is None:
        stored_config_path.write_text(resolved_config, encoding="utf-8")
    elif stored_config_path.read_text(encoding="utf-8") != resolved_config:
        raise ValueError("Resume configuration does not match the stored run configuration")
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    start_epoch = 1
    provenance_history: list[dict[str, object]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint.get("history", [])
        best_loss = float(checkpoint.get("best_loss", checkpoint["metrics"]["loss"]))
        start_epoch = int(checkpoint["epoch"]) + 1
        if "random_state" in checkpoint:
            restore_random_state(checkpoint["random_state"])
        stored_provenance = checkpoint.get("source_provenance", [])
        if isinstance(stored_provenance, list):
            provenance_history.extend(stored_provenance)
    provenance_history.append(launch_provenance)
    if start_epoch > int(training["epochs"]):
        raise ValueError("Checkpoint has already reached the configured epoch count")
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            training_loss_function,
            device,
            use_amp=bool(training.get("amp", True)),
            gradient_clip_norm=training.get("gradient_clip_norm"),
            geometry_loss_weight=float(training.get("geometry_loss_weight", 0.0)),
        )
        metrics = evaluate(
            model,
            val_loader,
            validation_loss_function,
            device,
            use_amp=bool(training.get("amp", True)),
            segmentation_threshold=float(evaluation["segmentation_threshold"]),
            detection_threshold=float(evaluation["detection_threshold"]),
            detection_minimum_distance=int(evaluation["detection_minimum_distance"]),
            detection_matching_radius=float(evaluation["detection_matching_radius"]),
        )
        scheduler.step()
        epoch_record: dict[str, float | int] = {"epoch": epoch, **train_metrics}
        epoch_record.update(metrics)
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "metrics": metrics,
            "history": history,
            "best_loss": min(best_loss, metrics["loss"]),
            "random_state": random_state(),
            "source_provenance": provenance_history,
        }
        save_checkpoint(checkpoint, run_dir / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(checkpoint, run_dir / "best.pt")

    summary = {
        "run_name": run_name,
        "status": "completed",
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": seed,
        "epochs": int(training["epochs"]),
        "duration_seconds": time.time() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "manifest_sha256": manifest_sha256(config["data"]["manifest"]),
        "git_commit": launch_provenance["git_commit"],
        "git_dirty": launch_provenance["git_dirty"],
        "source_provenance": provenance_history,
        "end_git_commit": git_commit(),
        "end_git_dirty": git_dirty(),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else None,
        "best_validation_loss": best_loss,
        "final_metrics": history[-1],
        "history": history,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
