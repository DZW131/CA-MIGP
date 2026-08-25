from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from bootstrap_ci import STAT_COLUMNS, STAT_INDEX  # noqa: E402
from bootstrap_compare_multiseed import (  # noqa: E402
    hierarchical_paired_samples,
    summarize,
)


def statistics(dice: float, pq: float) -> np.ndarray:
    value = np.zeros((1, len(STAT_COLUMNS)), dtype=float)
    value[0, STAT_INDEX["segmentation_dice_sum"]] = dice
    value[0, STAT_INDEX["segmentation_count"]] = 1.0
    value[0, STAT_INDEX["pq_sum"]] = pq
    value[0, STAT_INDEX["aji_plus_sum"]] = pq
    value[0, STAT_INDEX["instance_count"]] = 1.0
    return value


def test_hierarchical_bootstrap_preserves_constant_paired_effect() -> None:
    candidate = [statistics(0.8, 0.6) for _ in range(3)]
    reference = [statistics(0.5, 0.2) for _ in range(3)]
    samples = hierarchical_paired_samples(
        candidate,
        reference,
        replicates=101,
        rng=np.random.default_rng(2026),
    )
    summary = summarize(candidate, reference, samples, [17, 23, 42])

    assert samples["segmentation_dice"] == pytest.approx(np.full(101, 0.3))
    assert summary["mean_pq"]["estimate"] == pytest.approx(0.4)
    assert summary["mean_pq"]["ci95_low"] == pytest.approx(0.4)
    assert summary["mean_pq"]["probability_candidate_better"] == 1.0
