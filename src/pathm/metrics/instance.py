from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _contingency(
    reference: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference.shape != prediction.shape or reference.ndim != 2:
        raise ValueError("Instance maps must be identically shaped 2D arrays")
    reference_ids, reference_inverse = np.unique(reference, return_inverse=True)
    prediction_ids, prediction_inverse = np.unique(prediction, return_inverse=True)
    matrix = np.bincount(
        (reference_inverse * prediction_ids.size + prediction_inverse).ravel(),
        minlength=reference_ids.size * prediction_ids.size,
    ).reshape(reference_ids.size, prediction_ids.size)
    reference_areas = matrix.sum(axis=1)
    prediction_areas = matrix.sum(axis=0)
    reference_keep = reference_ids != 0
    prediction_keep = prediction_ids != 0
    intersections = matrix[np.ix_(reference_keep, prediction_keep)].astype(np.float64)
    return intersections, reference_areas[reference_keep], prediction_areas[prediction_keep]


def instance_scores(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    intersections, reference_areas, prediction_areas = _contingency(reference, prediction)
    if intersections.size:
        unions = reference_areas[:, None] + prediction_areas[None, :] - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > 0,
        )
    else:
        ious = np.zeros((reference_areas.size, prediction_areas.size), dtype=np.float64)

    matched_reference: np.ndarray
    matched_prediction: np.ndarray
    if ious.size:
        matched_reference, matched_prediction = linear_sum_assignment(-ious)
        positive = intersections[matched_reference, matched_prediction] > 0
        matched_reference = matched_reference[positive]
        matched_prediction = matched_prediction[positive]
    else:
        matched_reference = np.array([], dtype=np.int64)
        matched_prediction = np.array([], dtype=np.int64)

    matched_intersection = intersections[matched_reference, matched_prediction].sum()
    matched_union = (
        reference_areas[matched_reference]
        + prediction_areas[matched_prediction]
        - intersections[matched_reference, matched_prediction]
    ).sum()
    unmatched_reference = np.delete(reference_areas, matched_reference).sum()
    unmatched_prediction = np.delete(prediction_areas, matched_prediction).sum()
    aji_denominator = matched_union + unmatched_reference + unmatched_prediction
    aji_plus = matched_intersection / aji_denominator if aji_denominator else 1.0

    pq_matches = ious > 0.5
    true_positive = int(pq_matches.sum())
    false_positive = int(prediction_areas.size - true_positive)
    false_negative = int(reference_areas.size - true_positive)
    detection_quality_denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    detection_quality = (
        true_positive / detection_quality_denominator if detection_quality_denominator else 1.0
    )
    segmentation_quality = float(ious[pq_matches].mean()) if true_positive else 0.0
    return {
        "aji_plus": float(aji_plus),
        "pq": detection_quality * segmentation_quality,
        "dq": detection_quality,
        "sq": segmentation_quality,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }
