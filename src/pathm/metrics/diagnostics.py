from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import ndimage


def _instance_boundaries(instance_map: np.ndarray) -> np.ndarray:
    padded = np.pad(instance_map, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    boundary = np.zeros_like(center, dtype=bool)
    for neighbor in (
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
    ):
        boundary |= (center > 0) & (neighbor != center)
    return boundary


def boundary_f1(
    reference: np.ndarray,
    prediction: np.ndarray,
    tolerance: float = 2.0,
) -> float:
    """Compute symmetric foreground-boundary F1 within a pixel tolerance."""
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Instance maps must be identically shaped 2D arrays")
    if tolerance < 0.0:
        raise ValueError("Boundary tolerance must be non-negative")
    reference_boundary = _instance_boundaries(reference)
    prediction_boundary = _instance_boundaries(prediction)
    if not reference_boundary.any() and not prediction_boundary.any():
        return 1.0
    if not reference_boundary.any() or not prediction_boundary.any():
        return 0.0
    distance_to_reference = ndimage.distance_transform_edt(~reference_boundary)
    distance_to_prediction = ndimage.distance_transform_edt(~prediction_boundary)
    precision = float((distance_to_reference[prediction_boundary] <= tolerance).mean())
    recall = float((distance_to_prediction[reference_boundary] <= tolerance).mean())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def touching_instance_pairs(
    instance_map: np.ndarray,
    radius: float = 3.0,
) -> set[tuple[int, int]]:
    """Find reference-instance pairs separated by at most ``radius`` pixels."""
    if instance_map.ndim != 2:
        raise ValueError("Instance map must be two-dimensional")
    if radius <= 0.0:
        raise ValueError("Contact radius must be positive")
    pairs: set[tuple[int, int]] = set()
    padding = int(np.ceil(radius)) + 1
    objects = ndimage.find_objects(instance_map)
    height, width = instance_map.shape
    for instance_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        rows, columns = bounds
        top = max((rows.start or 0) - padding, 0)
        bottom = min((rows.stop or height) + padding, height)
        left = max((columns.start or 0) - padding, 0)
        right = min((columns.stop or width) + padding, width)
        local_labels = instance_map[top:bottom, left:right]
        local_instance = local_labels == instance_id
        distance = ndimage.distance_transform_edt(~local_instance)
        neighbors = np.unique(
            local_labels[(distance <= radius) & (local_labels != instance_id)]
        )
        for neighbor in neighbors:
            if neighbor != 0:
                pairs.add(tuple(sorted((instance_id, int(neighbor)))))
    return pairs


def instance_diagnostic_scores(
    reference: np.ndarray,
    prediction: np.ndarray,
    contact_radius: float = 3.0,
    boundary_tolerance: float = 2.0,
) -> dict[str, float]:
    """Measure boundary accuracy, counts, and separation of touching nuclei."""
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Instance maps must be identically shaped 2D arrays")
    reference_ids, reference_inverse = np.unique(reference, return_inverse=True)
    prediction_ids, prediction_inverse = np.unique(prediction, return_inverse=True)
    contingency = np.bincount(
        (reference_inverse * prediction_ids.size + prediction_inverse).ravel(),
        minlength=reference_ids.size * prediction_ids.size,
    ).reshape(reference_ids.size, prediction_ids.size)
    reference_keep = reference_ids != 0
    prediction_keep = prediction_ids != 0
    kept_reference_ids = reference_ids[reference_keep]
    kept_prediction_ids = prediction_ids[prediction_keep]
    intersections = contingency[np.ix_(reference_keep, prediction_keep)].astype(float)
    reference_areas = contingency[reference_keep].sum(axis=1).astype(float)
    prediction_areas = contingency[:, prediction_keep].sum(axis=0).astype(float)
    if intersections.size:
        unions = reference_areas[:, None] + prediction_areas[None, :] - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > 0,
        )
        best_prediction_indices = intersections.argmax(axis=1)
        best_intersections = intersections[
            np.arange(intersections.shape[0]), best_prediction_indices
        ]
        best_prediction_ids = kept_prediction_ids[best_prediction_indices]
        best_ious = ious[np.arange(ious.shape[0]), best_prediction_indices]
        best_prediction_ids = np.where(best_intersections > 0, best_prediction_ids, 0)
    else:
        best_prediction_ids = np.zeros(kept_reference_ids.size, dtype=int)
        best_ious = np.zeros(kept_reference_ids.size, dtype=float)

    id_to_index = {
        int(instance_id): index for index, instance_id in enumerate(kept_reference_ids)
    }
    contact_pairs = touching_instance_pairs(reference, radius=contact_radius)
    contact_ids = sorted({instance_id for pair in contact_pairs for instance_id in pair})
    contact_indices = [id_to_index[instance_id] for instance_id in contact_ids]
    contact_detected = sum(best_ious[index] > 0.5 for index in contact_indices)
    pair_successes = 0
    pair_merges = 0
    for first_id, second_id in contact_pairs:
        first_index = id_to_index[first_id]
        second_index = id_to_index[second_id]
        first_prediction = int(best_prediction_ids[first_index])
        second_prediction = int(best_prediction_ids[second_index])
        if first_prediction and first_prediction == second_prediction:
            pair_merges += 1
        if (
            best_ious[first_index] > 0.5
            and best_ious[second_index] > 0.5
            and first_prediction != second_prediction
        ):
            pair_successes += 1

    return {
        "boundary_f1": boundary_f1(
            reference,
            prediction,
            tolerance=boundary_tolerance,
        ),
        "reference_instances": float(kept_reference_ids.size),
        "predicted_instances": float(kept_prediction_ids.size),
        "count_absolute_error": float(
            abs(kept_prediction_ids.size - kept_reference_ids.size)
        ),
        "contact_instances": float(len(contact_ids)),
        "contact_instances_detected": float(contact_detected),
        "contact_pairs": float(len(contact_pairs)),
        "contact_pair_successes": float(pair_successes),
        "contact_pair_merges": float(pair_merges),
    }


def nucleus_count_weighted_mean(
    records: Sequence[Mapping[str, object]],
    metric: str,
) -> float:
    """Average an image metric using the reference nucleus count as its weight."""
    if not records:
        raise ValueError("Cannot aggregate an empty record sequence")
    weighted_sum = 0.0
    total_nuclei = 0.0
    for index, record in enumerate(records):
        value = float(record[metric])
        if "reference_instances" in record:
            weight = float(record["reference_instances"])
        elif "true_positive" in record and "false_negative" in record:
            weight = float(record["true_positive"]) + float(record["false_negative"])
        else:
            raise ValueError(f"Missing reference nucleus count in record {index}")
        if not math.isfinite(value) or not math.isfinite(weight):
            raise ValueError(f"Non-finite value in record {index}")
        if weight < 0.0:
            raise ValueError(f"Negative reference nucleus count in record {index}")
        weighted_sum += value * weight
        total_nuclei += weight
    if total_nuclei <= 0.0:
        raise ValueError("Reference nucleus count must be positive")
    return weighted_sum / total_nuclei
