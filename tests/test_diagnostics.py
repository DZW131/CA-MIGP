import numpy as np
import pytest

from pathm.metrics import (
    boundary_f1,
    instance_diagnostic_scores,
    nucleus_count_weighted_mean,
    touching_instance_pairs,
)


def _touching_reference() -> np.ndarray:
    reference = np.zeros((20, 20), dtype=np.uint16)
    reference[4:12, 2:8] = 1
    reference[4:12, 9:15] = 2
    reference[14:18, 14:18] = 3
    return reference


def test_touching_instance_pairs_respect_radius() -> None:
    reference = _touching_reference()

    assert touching_instance_pairs(reference, radius=2.0) == {(1, 2)}
    assert touching_instance_pairs(reference, radius=1.0) == set()


def test_diagnostics_detect_contact_pair_success() -> None:
    reference = _touching_reference()

    scores = instance_diagnostic_scores(reference, reference.copy(), contact_radius=2.0)

    assert scores["boundary_f1"] == 1.0
    assert scores["count_absolute_error"] == 0.0
    assert scores["contact_instances"] == 2.0
    assert scores["contact_instances_detected"] == 2.0
    assert scores["contact_pairs"] == 1.0
    assert scores["contact_pair_successes"] == 1.0
    assert scores["contact_pair_merges"] == 0.0


def test_diagnostics_detect_merged_contact_pair() -> None:
    reference = _touching_reference()
    prediction = reference.copy()
    prediction[prediction == 2] = 1
    prediction[4:12, 8] = 1

    scores = instance_diagnostic_scores(reference, prediction, contact_radius=2.0)

    assert scores["predicted_instances"] == 2.0
    assert scores["count_absolute_error"] == 1.0
    assert scores["contact_pair_successes"] == 0.0
    assert scores["contact_pair_merges"] == 1.0
    assert boundary_f1(reference, prediction, tolerance=0.0) < 1.0


def test_boundary_f1_handles_empty_maps() -> None:
    empty = np.zeros((8, 8), dtype=np.uint16)
    nonempty = empty.copy()
    nonempty[2:5, 2:5] = 1

    assert boundary_f1(empty, empty) == 1.0
    assert boundary_f1(empty, nonempty) == 0.0


def test_nucleus_count_weighted_mean_uses_reference_counts() -> None:
    records = [
        {"pq": 0.2, "reference_instances": 1},
        {"pq": 0.8, "reference_instances": 3},
    ]

    assert nucleus_count_weighted_mean(records, "pq") == pytest.approx(0.65)


def test_nucleus_count_weighted_mean_recovers_legacy_reference_counts() -> None:
    records = [
        {"pq": 0.2, "true_positive": 1, "false_negative": 0},
        {"pq": 0.8, "true_positive": 2, "false_negative": 1},
    ]

    assert nucleus_count_weighted_mean(records, "pq") == pytest.approx(0.65)
