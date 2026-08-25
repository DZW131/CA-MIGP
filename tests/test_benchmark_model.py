from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import benchmark_model as benchmark_module  # noqa: E402
from benchmark_model import active_parameter_count, gpu_process_snapshot  # noqa: E402
from pathm.models import PathMUnified  # noqa: E402


def test_active_parameter_count_excludes_training_only_geometry_branch() -> None:
    model = PathMUnified(
        base_channels=8,
        prompt_dim=16,
        style_channels=4,
        use_geometry_prototypes=True,
    ).eval()
    images = torch.rand(1, 3, 32, 32)
    operation_ids = torch.zeros(1, dtype=torch.long)
    target_ids = torch.zeros(1, dtype=torch.long)

    active = active_parameter_count([model], lambda: model(images, operation_ids, target_ids))
    total = sum(parameter.numel() for parameter in model.parameters())

    assert 0 < active < total
    assert total - active == sum(
        parameter.numel()
        for parameter in model.geometry_prototypes.parameters()  # type: ignore[union-attr]
    )


def test_gpu_process_snapshot_parses_nvidia_smi_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_module.subprocess,
        "check_output",
        lambda *args, **kwargs: "123, python, 6400\n456, worker, 128.5\n",
    )

    processes = gpu_process_snapshot()

    assert processes == [
        {"pid": 123, "process_name": "python", "used_gpu_memory_mib": 6400.0},
        {"pid": 456, "process_name": "worker", "used_gpu_memory_mib": 128.5},
    ]
