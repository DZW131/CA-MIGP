from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.models import build_model  # noqa: E402
from pathm.training import MixedTaskLoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_complete.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int)
    return parser.parse_args()


def synthetic_batch(batch_size: int, image_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    images = torch.rand(batch_size, 3, image_size, image_size, device=device) * 0.15
    targets = torch.zeros(batch_size, 1, image_size, image_size, device=device)
    operation_ids = torch.arange(batch_size, device=device) % 2
    target_ids = torch.arange(batch_size, device=device) % 3
    yy, xx = torch.meshgrid(
        torch.arange(image_size, device=device),
        torch.arange(image_size, device=device),
        indexing="ij",
    )
    for index in range(batch_size):
        center_x = image_size // 3 + index * 3
        center_y = image_size // 2 - index * 2
        radius = 3 + index
        if operation_ids[index].item() == 0:
            gaussian = torch.exp(
                -((xx - center_x).square() + (yy - center_y).square()) / (2.0 * radius**2)
            )
            targets[index, 0] = gaussian
        else:
            mask = (xx - center_x).square() + (yy - center_y).square() <= (radius * 2) ** 2
            targets[index, 0] = mask.float()
        images[index, 0] += targets[index, 0] * 0.75
        images[index, 1] += targets[index, 0] * 0.35
    return images.clamp_(0, 1), targets, operation_ids, target_ids


def main() -> None:
    args = parse_args()
    with (REPOSITORY_ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    model = build_model(config["model"]).to(device)
    loss_function = MixedTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    steps = args.steps or config["training"]["steps"]
    losses: list[float] = []

    model.train()
    for step in range(steps):
        batch = synthetic_batch(
            config["training"]["batch_size"],
            config["training"]["image_size"],
            device,
        )
        images, targets, operation_ids, target_ids = batch
        optimizer.zero_grad(set_to_none=True)
        if config["model"].get("use_geometry_prototypes", False):
            prototype_count = int(config["model"].get("geometry_num_prototypes", 5))
            geometry_masks = torch.zeros(
                images.shape[0],
                prototype_count,
                images.shape[-2],
                images.shape[-1],
                dtype=torch.bool,
                device=device,
            )
            for batch_index in torch.nonzero(operation_ids == 1).flatten().tolist():
                for prototype_index in range(prototype_count):
                    start = 2 + prototype_index * 3
                    geometry_masks[
                        batch_index,
                        prototype_index,
                        start : start + 2,
                        start : start + 2,
                    ] = True
            outputs = model(
                images,
                operation_ids,
                target_ids,
                geometry_masks=geometry_masks,
            )
        else:
            outputs = model(images, operation_ids, target_ids)
        loss = loss_function(outputs["logits"], targets, operation_ids)
        loss = loss + float(config["training"].get("geometry_loss_weight", 0.0)) * outputs.get(
            "geometry_loss", loss.new_zeros(())
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"step={step + 1:03d} loss={losses[-1]:.6f}")

    if not all(np.isfinite(losses)):
        raise RuntimeError("Smoke test produced a non-finite loss")
    print(
        "smoke_ok",
        f"device={device}",
        f"initial_loss={losses[0]:.6f}",
        f"final_loss={losses[-1]:.6f}",
    )


if __name__ == "__main__":
    main()
