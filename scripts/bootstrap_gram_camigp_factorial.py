from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from bootstrap_compare_multiseed import load_seed_statistics  # noqa: E402
from bootstrap_gram_gsc_factorial import (  # noqa: E402
    CELLS,
    factorial_hierarchical_samples,
    factorial_point_summary,
)


DEFAULT_STEMS = {
    "base": "pannuke_ablation_no_style",
    "gram": "pannuke_full_prompt",
    "gsc": "pannuke_no_style_geometry",
    "combined": "pannuke_geometry_no_orth",
}
EFFECT_RENAMES = {
    "dynamic_gram_without_gsc": "dynamic_gram_without_camigp",
    "dynamic_gram_with_gsc": "dynamic_gram_with_camigp",
    "gsc_without_dynamic_gram": "camigp_without_dynamic_gram",
    "gsc_with_dynamic_gram": "camigp_with_dynamic_gram",
    "gsc_main_effect": "camigp_main_effect",
    "dynamic_gram_by_gsc_interaction": "dynamic_gram_by_camigp_interaction",
}


@dataclass(frozen=True)
class DatasetSpec:
    relative_dir: str
    cluster_key: str
    task_file: str
    instance_file: str


def parse_dataset_spec(value: str) -> DatasetSpec:
    parts = value.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "dataset specs must be RELATIVE_DIR:CLUSTER_KEY:TASK_FILE:INSTANCE_FILE"
        )
    relative_dir, cluster_key, task_file, instance_file = parts
    if not all(parts):
        raise argparse.ArgumentTypeError("dataset-spec fields must be non-empty")
    if cluster_key not in {"domain", "sample_id"}:
        raise argparse.ArgumentTypeError("cluster key must be domain or sample_id")
    return DatasetSpec(relative_dir, cluster_key, task_file, instance_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical paired bootstrap for Dynamic-Gram x CA-MIGP"
    )
    parser.add_argument("--outputs-root", required=True, type=Path)
    for cell in CELLS:
        option = "camigp" if cell == "gsc" else cell
        parser.add_argument(f"--{option}-stem", default=DEFAULT_STEMS[cell])
    parser.add_argument(
        "--dataset-spec",
        required=True,
        action="append",
        type=parse_dataset_spec,
        help="RELATIVE_DIR:CLUSTER_KEY:TASK_FILE:INSTANCE_FILE; repeat per dataset",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23, 42))
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def rename_factorial_summary(
    summary: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for metric, values in summary.items():
        renamed: dict[str, object] = {}
        for key, value in values.items():
            if key == "cell_equal_seed_dataset_means":
                if not isinstance(value, dict):
                    raise ValueError("Invalid factorial cell summary")
                renamed[key] = {
                    ("camigp" if cell == "gsc" else cell): cell_value
                    for cell, cell_value in value.items()
                }
            else:
                renamed[EFFECT_RENAMES.get(key, key)] = value
        output[metric] = renamed
    return output


def main() -> None:
    args = parse_args()
    if not args.seeds or not args.dataset_spec:
        raise ValueError("Seeds and dataset specifications must be non-empty")
    stems = {
        "base": args.base_stem,
        "gram": args.gram_stem,
        "gsc": args.camigp_stem,
        "combined": args.combined_stem,
    }
    cells: dict[str, list[list[np.ndarray]]] = {cell: [] for cell in CELLS}
    groups: dict[str, dict[str, dict[str, int | str]]] = {}
    dataset_labels = [spec.relative_dir for spec in args.dataset_spec]
    if len(set(dataset_labels)) != len(dataset_labels):
        raise ValueError("Dataset relative directories must be unique")

    for seed in args.seeds:
        groups[str(seed)] = {}
        for cell in CELLS:
            cells[cell].append([])
        for spec in args.dataset_spec:
            expected_groups: list[str] | None = None
            for cell in CELLS:
                group_ids, statistics = load_seed_statistics(
                    args.outputs_root,
                    stems[cell],
                    seed,
                    spec.relative_dir,
                    spec.task_file,
                    spec.instance_file,
                    spec.cluster_key,
                )
                if expected_groups is None:
                    expected_groups = group_ids
                elif group_ids != expected_groups:
                    raise ValueError(
                        f"Factorial groups differ for seed {seed}, {spec.relative_dir}"
                    )
                cells[cell][-1].append(statistics)
            assert expected_groups is not None
            groups[str(seed)][spec.relative_dir] = {
                "cluster_key": spec.cluster_key,
                "groups": len(expected_groups),
            }

    samples = factorial_hierarchical_samples(
        cells,
        args.replicates,
        np.random.default_rng(args.seed),
    )
    raw_summary = factorial_point_summary(
        cells,
        samples,
        args.seeds,
        dataset_labels,
    )
    output = {
        "bootstrap_unit": "seed_then_dataset_then_dataset_specific_group",
        "cells": {
            "base": stems["base"],
            "gram": stems["gram"],
            "camigp": stems["gsc"],
            "combined": stems["combined"],
        },
        "dataset_specs": [
            {
                "relative_dir": spec.relative_dir,
                "cluster_key": spec.cluster_key,
                "task_file": spec.task_file,
                "instance_file": spec.instance_file,
            }
            for spec in args.dataset_spec
        ],
        "dataset_weighting": "equal",
        "difference": "dynamic_gram_by_camigp_factorial",
        "groups": groups,
        "metrics": rename_factorial_summary(raw_summary),
        "replicates": args.replicates,
        "seed": args.seed,
        "seeds": args.seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
