from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_nuinsseg_stacked_masks import audit_stacked_masks  # noqa: E402
from evaluate_instances import (  # noqa: E402
    load_nuinsseg_stacked_masks,
    nuinsseg_stacked_mask_path,
    summarize_nuceval_records,
)
from summarize_nuceval_sensitivity import (  # noqa: E402
    aggregate as aggregate_nuceval,
)
from summarize_nuceval_sensitivity import collect as collect_nuceval  # noqa: E402
from summarize_nuceval_sensitivity import (  # noqa: E402
    paired_differences as nuceval_differences,
)


def test_nuinsseg_stacked_mask_path_maps_processed_image() -> None:
    path = nuinsseg_stacked_mask_path(
        Path("/raw/nuinsseg"),
        "NuInsSeg:human bladder",
        "/processed/human_bladder_01.png",
    )

    assert path == Path("/raw/nuinsseg/extracted/human bladder/stacked mask/human_bladder_01.mat")


def test_load_nuinsseg_stacked_masks_preserves_overlaps_and_removes_empty(
    tmp_path: Path,
) -> None:
    source = tmp_path / "extracted" / "human bladder" / "stacked mask" / "human_bladder_01.mat"
    source.parent.mkdir(parents=True)
    stack = np.zeros((4, 5, 3), dtype=np.uint8)
    stack[1:3, 1:3, 0] = 1
    stack[2:4, 2:4, 1] = 1
    savemat(source, {"stacked_mask": stack})

    masks, resolved_source, empty_removed = load_nuinsseg_stacked_masks(
        tmp_path,
        "NuInsSeg:human bladder",
        "/processed/human_bladder_01.png",
        (4, 5),
    )

    assert resolved_source == source
    assert len(masks) == 2
    assert empty_removed == 1
    assert np.count_nonzero(masks[0] & masks[1]) == 1


def test_load_nuinsseg_stacked_masks_restores_singleton_axis(tmp_path: Path) -> None:
    source = tmp_path / "extracted" / "human bladder" / "stacked mask" / "human_bladder_01.mat"
    source.parent.mkdir(parents=True)
    singleton = np.zeros((4, 5), dtype=np.uint8)
    singleton[1:3, 1:3] = 1
    savemat(source, {"stacked_mask": singleton})

    masks, _, empty_removed = load_nuinsseg_stacked_masks(
        tmp_path,
        "NuInsSeg:human bladder",
        "/processed/human_bladder_01.png",
        (4, 5),
    )

    assert len(masks) == 1
    assert empty_removed == 0


def test_summarize_nuceval_records_reports_both_aggregation_modes() -> None:
    records = []
    for pq, nuclei in ((0.2, 1), (0.8, 3)):
        records.append(
            {
                "zone_width": 0,
                "reference_instances": nuclei,
                "dice": pq,
                "aji": pq,
                "dq": pq,
                "sq": pq,
                "pq": pq,
            }
        )

    summary = summarize_nuceval_records(records)["0"]

    assert summary["unweighted"]["pq"] == pytest.approx(0.5)
    assert summary["nucleus_count_weighted"]["pq"] == pytest.approx(0.65)
    assert summary["reference_nuclei"] == 4


def test_audit_stacked_masks_checks_manifest_and_foreground_union(tmp_path: Path) -> None:
    source = (
        tmp_path / "raw" / "extracted" / "human bladder" / "stacked mask" / "human_bladder_01.mat"
    )
    binary_path = (
        tmp_path / "raw" / "extracted" / "human bladder" / "mask binary" / "human_bladder_01.png"
    )
    source.parent.mkdir(parents=True)
    binary_path.parent.mkdir(parents=True)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    savemat(source, {"stacked_mask": mask})
    Image.fromarray(mask * 255).save(binary_path)
    manifest = tmp_path / "nuinsseg.jsonl"
    manifest.write_text(
        '{"domain":"NuInsSeg:human bladder","image":"/processed/human_bladder_01.png",'
        '"operation":"segmentation"}\n',
        encoding="utf-8",
    )

    result = audit_stacked_masks(
        manifest,
        tmp_path / "raw",
        expected_images=1,
        expected_domains=1,
        expected_singleton_tensors=1,
        expected_empty_slices=0,
    )

    assert result["status"] == "passed"
    assert result["observed"]["nonempty_instances"] == 1
    assert result["foreground_union_mismatch_images"] == 0


def test_summarize_nuceval_sensitivity_requires_complete_paired_runs(
    tmp_path: Path,
) -> None:
    source_hash = "54ddb315af7547a7f465ae2d1208e16949669c913a164e3a80e59c9898180788"
    for family, stem in (
        ("full", "pannuke_full_prompt"),
        ("camigp", "pannuke_geometry_no_orth"),
    ):
        for seed_index, seed in enumerate((17, 23, 42)):
            value = 0.3 + seed_index * 0.01 + (0.1 if family == "camigp" else 0.0)
            metrics = {metric: value for metric in ("dice", "aji", "dq", "sq", "pq")}
            payload = {
                "nuceval_sha256": source_hash,
                "results": {
                    str(zone): {
                        "images": 665,
                        "unweighted": metrics,
                        "nucleus_count_weighted": metrics,
                    }
                    for zone in (0, 1)
                },
            }
            path = (
                tmp_path
                / f"{stem}_seed{seed}"
                / "nuinsseg_nuceval"
                / "ood_test_nuceval_sensitivity_summary.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

    records = collect_nuceval(tmp_path)
    aggregates = aggregate_nuceval(records)
    differences = nuceval_differences(records)

    assert len(records) == 120
    assert len(aggregates) == 40
    assert len(differences) == 20
    assert all(row["mean_difference"] == pytest.approx(0.1) for row in differences)
