from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


Point = tuple[float, float]


def heatmap_points(
    heatmap: Tensor,
    threshold: float = 0.5,
    minimum_distance: int = 3,
) -> list[Point]:
    """Extract local maxima as ``(row, column)`` points from one 2D heatmap."""
    if heatmap.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {tuple(heatmap.shape)}")
    if minimum_distance < 1:
        raise ValueError("minimum_distance must be at least 1")
    kernel_size = 2 * minimum_distance + 1
    values = heatmap[None, None]
    maxima = F.max_pool2d(values, kernel_size, stride=1, padding=minimum_distance)
    peak_mask = (values >= threshold) & torch.isclose(values, maxima)
    coordinates = torch.nonzero(peak_mask[0, 0], as_tuple=False)
    return [(float(row), float(column)) for row, column in coordinates.tolist()]


def point_detection_scores(
    predicted: Sequence[Point],
    reference: Sequence[Point],
    matching_radius: float,
) -> dict[str, float]:
    """Compute one-to-one point matching scores using deterministic greedy assignment."""
    candidates: list[tuple[float, int, int]] = []
    for predicted_index, predicted_point in enumerate(predicted):
        for reference_index, reference_point in enumerate(reference):
            distance = math.dist(predicted_point, reference_point)
            if distance <= matching_radius:
                candidates.append((distance, predicted_index, reference_index))
    candidates.sort()

    used_predicted: set[int] = set()
    used_reference: set[int] = set()
    for _, predicted_index, reference_index in candidates:
        if predicted_index in used_predicted or reference_index in used_reference:
            continue
        used_predicted.add(predicted_index)
        used_reference.add(reference_index)

    true_positive = len(used_predicted)
    false_positive = len(predicted) - true_positive
    false_negative = len(reference) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }
