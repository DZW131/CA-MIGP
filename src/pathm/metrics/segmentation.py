from __future__ import annotations

from torch import Tensor


def binary_segmentation_scores(
    prediction: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
    valid_mask: Tensor | None = None,
) -> dict[str, float]:
    predicted = prediction >= threshold
    reference = target >= threshold
    if valid_mask is not None:
        if valid_mask.shape != reference.shape:
            raise ValueError("valid_mask must match the prediction shape")
        predicted = predicted & valid_mask
        reference = reference & valid_mask
    intersection = (predicted & reference).sum().item()
    predicted_count = predicted.sum().item()
    reference_count = reference.sum().item()
    union = (predicted | reference).sum().item()
    return {
        "dice": (2.0 * intersection + epsilon)
        / (predicted_count + reference_count + epsilon),
        "iou": (intersection + epsilon) / (union + epsilon),
    }
