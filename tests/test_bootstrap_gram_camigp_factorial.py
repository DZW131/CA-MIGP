from __future__ import annotations

import argparse

import pytest

from scripts.bootstrap_gram_camigp_factorial import (
    parse_dataset_spec,
    rename_factorial_summary,
)


def test_dataset_spec_parses_dataset_specific_grouping() -> None:
    spec = parse_dataset_spec(
        "nuinsseg_ood:domain:per_image_tasks.jsonl:ood_test_instances.jsonl"
    )
    assert spec.relative_dir == "nuinsseg_ood"
    assert spec.cluster_key == "domain"
    assert spec.task_file == "per_image_tasks.jsonl"
    assert spec.instance_file == "ood_test_instances.jsonl"


@pytest.mark.parametrize(
    "value",
    (
        "nuinsseg_ood:patient:tasks.jsonl:instances.jsonl",
        "nuinsseg_ood:domain:tasks.jsonl",
        ":domain:tasks.jsonl:instances.jsonl",
    ),
)
def test_dataset_spec_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_dataset_spec(value)


def test_factorial_summary_uses_camigp_names_without_changing_values() -> None:
    raw = {
        "mean_pq": {
            "cell_equal_seed_dataset_means": {
                "base": 0.1,
                "gram": 0.2,
                "gsc": 0.3,
                "combined": 0.4,
            },
            "dynamic_gram_without_gsc": {"estimate": 0.1},
            "dynamic_gram_with_gsc": {"estimate": 0.1},
            "gsc_without_dynamic_gram": {"estimate": 0.2},
            "gsc_with_dynamic_gram": {"estimate": 0.2},
            "dynamic_gram_main_effect": {"estimate": 0.1},
            "gsc_main_effect": {"estimate": 0.2},
            "dynamic_gram_by_gsc_interaction": {"estimate": 0.0},
            "complete_method_vs_base": {"estimate": 0.3},
        }
    }
    renamed = rename_factorial_summary(raw)["mean_pq"]
    assert renamed["cell_equal_seed_dataset_means"] == {
        "base": 0.1,
        "gram": 0.2,
        "camigp": 0.3,
        "combined": 0.4,
    }
    assert renamed["camigp_main_effect"] == {"estimate": 0.2}
    assert renamed["dynamic_gram_by_camigp_interaction"] == {"estimate": 0.0}
    assert not any("gsc" in key for key in renamed)
