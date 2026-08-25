from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from pathm.metrics import binary_segmentation_scores, heatmap_points, point_detection_scores

from .losses import MixedTaskLoss


def _to_device(batch: dict[str, object], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["image"].to(device, non_blocking=True),  # type: ignore[union-attr]
        batch["target"].to(device, non_blocking=True),  # type: ignore[union-attr]
        batch["operation_id"].to(device, non_blocking=True),  # type: ignore[union-attr]
        batch["target_id"].to(device, non_blocking=True),  # type: ignore[union-attr]
    )


def mismatched_style_images(images: torch.Tensor, sample_ids: list[str]) -> torch.Tensor:
    """Derange a batch so no sample receives its own style source."""
    if images.shape[0] != len(sample_ids):
        raise ValueError("Image batch and sample IDs must have the same length")
    if images.shape[0] < 2:
        raise ValueError("Style mismatch evaluation requires at least two samples")
    for shift in range(1, images.shape[0]):
        indices = torch.roll(torch.arange(images.shape[0], device=images.device), shift)
        if all(
            sample_ids[index] != sample_ids[int(indices[index])]
            for index in range(len(sample_ids))
        ):
            return images[indices]
    raise ValueError("Could not construct a style-source derangement for this batch")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: MixedTaskLoss,
    device: torch.device,
    use_amp: bool,
    gradient_clip_norm: float | None = None,
    geometry_loss_weight: float = 0.0,
) -> dict[str, float]:
    if geometry_loss_weight < 0.0:
        raise ValueError("geometry_loss_weight must be non-negative")
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_geometry_loss = 0.0
    total_geometry_samples = 0.0
    total_samples = 0
    for batch in loader:
        images, targets, operation_ids, target_ids = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            geometry_masks = (
                batch["geometry_masks"].to(device, non_blocking=True)  # type: ignore[union-attr]
                if "geometry_masks" in batch
                else None
            )
            if geometry_masks is None or geometry_loss_weight == 0.0:
                outputs = model(images, operation_ids, target_ids)
            else:
                outputs = model(
                    images,
                    operation_ids,
                    target_ids,
                    geometry_masks=geometry_masks,
                )
            task_loss = loss_function(
                outputs["logits"],
                targets,
                operation_ids,
                geometry_masks=geometry_masks,
            )
            geometry_loss = outputs.get("geometry_loss", task_loss.new_zeros(()))
            loss = task_loss + geometry_loss_weight * geometry_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss)}")
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_task_loss += float(task_loss.detach()) * batch_size
        total_geometry_loss += float(geometry_loss.detach()) * batch_size
        total_geometry_samples += float(
            outputs.get("geometry_sample_count", task_loss.new_zeros(())).detach()
        )
        total_samples += batch_size
    denominator = max(total_samples, 1)
    return {
        "train_loss": total_loss / denominator,
        "train_task_loss": total_task_loss / denominator,
        "train_geometry_loss": total_geometry_loss / denominator,
        "train_geometry_samples_per_batch": total_geometry_samples / max(len(loader), 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: MixedTaskLoss,
    device: torch.device,
    use_amp: bool,
    segmentation_threshold: float = 0.5,
    detection_threshold: float = 0.5,
    detection_minimum_distance: int = 3,
    detection_matching_radius: float = 6.0,
    per_image_records: list[dict[str, float | str]] | None = None,
    mismatch_style: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    segmentation_sums: defaultdict[str, float] = defaultdict(float)
    segmentation_count = 0
    detection_counts: defaultdict[str, float] = defaultdict(float)
    detection_maxima: list[float] = []

    for batch in loader:
        images, targets, operation_ids, target_ids = _to_device(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == "cuda",
        ):
            if mismatch_style:
                style_images = (
                    batch["style_image"].to(device, non_blocking=True)  # type: ignore[union-attr]
                    if "style_image" in batch
                    else mismatched_style_images(images, list(batch["sample_id"]))
                )
                logits = model(
                    images,
                    operation_ids,
                    target_ids,
                    style_images=style_images,
                )["logits"]
            else:
                logits = model(images, operation_ids, target_ids)["logits"]
            loss = loss_function(logits, targets, operation_ids)
        probabilities = torch.sigmoid(logits.float()).cpu()
        reference_targets = targets.float().cpu()
        valid_masks = batch["valid_mask"].cpu()
        total_loss += float(loss) * images.shape[0]
        total_samples += images.shape[0]

        for index, operation in enumerate(batch["operation"]):
            if operation == "segmentation":
                scores = binary_segmentation_scores(
                    probabilities[index, 0],
                    reference_targets[index, 0],
                    threshold=segmentation_threshold,
                    valid_mask=valid_masks[index, 0],
                )
                for key, value in scores.items():
                    segmentation_sums[key] += value
                segmentation_count += 1
                if per_image_records is not None:
                    record: dict[str, float | str] = {
                        "sample_id": batch["sample_id"][index],
                        "domain": batch["domain"][index],
                        "image_path": batch["image_path"][index],
                        "operation": operation,
                        **scores,
                    }
                    if mismatch_style and "style_source_domain" in batch:
                        record.update(
                            {
                                "image_path": batch["image_path"][index],
                                "style_source_sample_id": batch[
                                    "style_source_sample_id"
                                ][index],
                                "style_source_domain": batch["style_source_domain"][index],
                                "style_source_image": batch["style_source_image"][index],
                            }
                        )
                    per_image_records.append(record)
            else:
                detection_maxima.append(float(probabilities[index, 0].max()))
                predicted = heatmap_points(
                    probabilities[index, 0],
                    threshold=detection_threshold,
                    minimum_distance=detection_minimum_distance,
                )
                reference = heatmap_points(
                    reference_targets[index, 0],
                    threshold=0.99,
                    minimum_distance=detection_minimum_distance,
                )
                predicted = [
                    point
                    for point in predicted
                    if valid_masks[index, 0, int(point[0]), int(point[1])]
                ]
                reference = [
                    point
                    for point in reference
                    if valid_masks[index, 0, int(point[0]), int(point[1])]
                ]
                scores = point_detection_scores(
                    predicted,
                    reference,
                    matching_radius=detection_matching_radius,
                )
                for key in ("true_positive", "false_positive", "false_negative"):
                    detection_counts[key] += scores[key]
                if per_image_records is not None:
                    record = {
                        "sample_id": batch["sample_id"][index],
                        "domain": batch["domain"][index],
                        "image_path": batch["image_path"][index],
                        "operation": operation,
                        "max_probability": detection_maxima[-1],
                        **scores,
                    }
                    if mismatch_style and "style_source_domain" in batch:
                        record.update(
                            {
                                "image_path": batch["image_path"][index],
                                "style_source_sample_id": batch[
                                    "style_source_sample_id"
                                ][index],
                                "style_source_domain": batch["style_source_domain"][index],
                                "style_source_image": batch["style_source_image"][index],
                            }
                        )
                    per_image_records.append(record)

    metrics = {"loss": total_loss / max(total_samples, 1)}
    if segmentation_count:
        metrics.update(
            {
                f"segmentation_{key}": value / segmentation_count
                for key, value in segmentation_sums.items()
            }
        )
    if detection_counts:
        true_positive = detection_counts["true_positive"]
        false_positive = detection_counts["false_positive"]
        false_negative = detection_counts["false_negative"]
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        metrics.update(
            {
                "detection_precision": precision,
                "detection_recall": recall,
                "detection_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "detection_max_probability_mean": sum(detection_maxima)
                / len(detection_maxima),
                "detection_max_probability_max": max(detection_maxima),
            }
        )
    return metrics
