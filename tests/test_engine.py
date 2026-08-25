from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from pathm.models import MultiHeadUNet, PathMUnified
from pathm.training import MixedTaskLoss, train_one_epoch


def _sample(operation_id: int, with_geometry: bool) -> dict[str, object]:
    sample: dict[str, object] = {
        "image": torch.rand(3, 32, 32),
        "target": torch.rand(1, 32, 32),
        "operation_id": torch.tensor(operation_id),
        "target_id": torch.tensor(0),
    }
    if with_geometry:
        masks = torch.zeros(5, 32, 32, dtype=torch.bool)
        if operation_id == 1:
            for prototype_index in range(5):
                start = 2 + prototype_index * 4
                masks[prototype_index, start : start + 3, start : start + 3] = True
        sample["geometry_masks"] = masks
    return sample


def test_train_one_epoch_combines_geometry_loss() -> None:
    model = PathMUnified(
        base_channels=4,
        prompt_dim=16,
        style_channels=4,
        use_geometry_prototypes=True,
        geometry_max_samples_per_prototype=8,
    )
    loader = DataLoader([_sample(0, True), _sample(1, True)], batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        MixedTaskLoss(),
        torch.device("cpu"),
        use_amp=False,
        geometry_loss_weight=0.1,
    )

    assert metrics["train_geometry_loss"] > 0.0
    assert metrics["train_geometry_samples_per_batch"] > 0.0
    assert metrics["train_loss"] > metrics["train_task_loss"]


def test_train_one_epoch_keeps_non_geometry_baseline_compatible() -> None:
    model = MultiHeadUNet(base_channels=4)
    loader = DataLoader([_sample(0, False), _sample(1, False)], batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        MixedTaskLoss(),
        torch.device("cpu"),
        use_amp=False,
    )

    assert metrics["train_geometry_loss"] == 0.0
    assert metrics["train_geometry_samples_per_batch"] == 0.0
    assert metrics["train_loss"] == metrics["train_task_loss"]


def test_train_one_epoch_uses_geometry_masks_for_uncertainty_without_prototypes() -> None:
    model = PathMUnified(base_channels=4, prompt_dim=16, style_channels=4)
    loader = DataLoader([_sample(0, True), _sample(1, True)], batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        MixedTaskLoss(
            uncertainty_boundary_alpha=3.0,
            uncertainty_boundary_mode="state",
        ),
        torch.device("cpu"),
        use_amp=False,
    )

    assert torch.isfinite(torch.tensor(metrics["train_task_loss"]))
    assert metrics["train_geometry_loss"] == 0.0
