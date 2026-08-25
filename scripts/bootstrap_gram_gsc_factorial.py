from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bootstrap_ci import metrics_from_sufficient
from bootstrap_compare_multiseed import load_seed_statistics


CELLS = ("base", "gram", "gsc", "combined")
DEFAULT_STEMS = {
    "base": "pannuke_ablation_no_style",
    "gram": "pannuke_full_prompt",
    "gsc": "pannuke_no_style_gsc_ube",
    "combined": "pannuke_gsc_ube",
}
EFFECTS = (
    "dynamic_gram_without_gsc",
    "dynamic_gram_with_gsc",
    "gsc_without_dynamic_gram",
    "gsc_with_dynamic_gram",
    "dynamic_gram_main_effect",
    "gsc_main_effect",
    "dynamic_gram_by_gsc_interaction",
    "complete_method_vs_base",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical paired bootstrap for the Dynamic-Gram x GSC design"
    )
    parser.add_argument("--outputs-root", required=True, type=Path)
    for cell in CELLS:
        parser.add_argument(f"--{cell}-stem", default=DEFAULT_STEMS[cell])
    parser.add_argument("--relative-dirs", required=True, nargs="+")
    parser.add_argument("--task-file", default="per_image_tasks.jsonl")
    parser.add_argument("--instance-file", default="ood_test_instances.jsonl")
    parser.add_argument("--cluster-key", choices=("domain", "sample_id"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23, 42))
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def factorial_effects(cells: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    if set(cells) != set(CELLS):
        raise ValueError(f"Expected factorial cells {CELLS}, received {sorted(cells)}")
    metrics = set(cells[CELLS[0]])
    if any(set(cells[cell]) != metrics for cell in CELLS[1:]):
        raise ValueError("Factorial cells contain different metrics")
    output: dict[str, dict[str, float]] = {}
    for metric in sorted(metrics):
        base = cells["base"][metric]
        gram = cells["gram"][metric]
        gsc = cells["gsc"][metric]
        combined = cells["combined"][metric]
        gram_without_gsc = gram - base
        gram_with_gsc = combined - gsc
        gsc_without_gram = gsc - base
        gsc_with_gram = combined - gram
        output[metric] = {
            "dynamic_gram_without_gsc": gram_without_gsc,
            "dynamic_gram_with_gsc": gram_with_gsc,
            "gsc_without_dynamic_gram": gsc_without_gram,
            "gsc_with_dynamic_gram": gsc_with_gram,
            "dynamic_gram_main_effect": 0.5 * (gram_without_gsc + gram_with_gsc),
            "gsc_main_effect": 0.5 * (gsc_without_gram + gsc_with_gram),
            "dynamic_gram_by_gsc_interaction": gram_with_gsc - gram_without_gsc,
            "complete_method_vs_base": combined - base,
        }
    return output


def validate_statistics(
    cells: dict[str, list[list[np.ndarray]]],
) -> tuple[int, int]:
    if set(cells) != set(CELLS):
        raise ValueError(f"Expected factorial cells {CELLS}, received {sorted(cells)}")
    seed_count = len(cells[CELLS[0]])
    if seed_count == 0 or any(len(cells[cell]) != seed_count for cell in CELLS):
        raise ValueError("Every factorial cell must contain the same non-zero seeds")
    dataset_count = len(cells[CELLS[0]][0])
    if dataset_count == 0 or any(
        len(seed) != dataset_count
        for cell in CELLS
        for seed in cells[cell]
    ):
        raise ValueError("Every seed must contain the same non-zero datasets")
    for seed_index in range(seed_count):
        for dataset_index in range(dataset_count):
            shapes = {
                cells[cell][seed_index][dataset_index].shape for cell in CELLS
            }
            if len(shapes) != 1:
                raise ValueError("Factorial group matrices are not paired")
    return seed_count, dataset_count


def factorial_hierarchical_samples(
    cells: dict[str, list[list[np.ndarray]]],
    replicates: int,
    rng: np.random.Generator,
    *,
    chunk_size: int = 128,
) -> dict[str, dict[str, np.ndarray]]:
    seed_count, dataset_count = validate_statistics(cells)
    if replicates <= 0:
        raise ValueError("Replicates must be positive")
    samples: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        chosen_seeds = rng.integers(0, seed_count, size=(size, seed_count))
        chosen_datasets = rng.integers(
            0, dataset_count, size=(size, seed_count, dataset_count)
        )
        chunk: dict[tuple[str, str], np.ndarray] = {}
        for seed_position in range(seed_count):
            for dataset_position in range(dataset_count):
                selections = zip(
                    chosen_seeds[:, seed_position],
                    chosen_datasets[:, seed_position, dataset_position],
                    strict=True,
                )
                pairs: dict[tuple[int, int], list[int]] = {}
                for replicate, pair in enumerate(selections):
                    pairs.setdefault(pair, []).append(replicate)
                for (seed_index, dataset_index), replicate_indices in pairs.items():
                    selected = np.asarray(replicate_indices, dtype=np.int64)
                    groups = cells["base"][seed_index][dataset_index].shape[0]
                    counts = rng.multinomial(
                        groups,
                        np.full(groups, 1.0 / groups),
                        size=selected.size,
                    )
                    sampled_cells = {
                        cell: metrics_from_sufficient(
                            counts @ cells[cell][seed_index][dataset_index]
                        )
                        for cell in CELLS
                    }
                    for metric, effects in factorial_effects(sampled_cells).items():
                        for effect, value in effects.items():
                            values = chunk.setdefault(
                                (metric, effect),
                                np.zeros(
                                    (size, seed_count, dataset_count),
                                    dtype=np.float64,
                                ),
                            )
                            values[selected, seed_position, dataset_position] = value
        for key, values in chunk.items():
            samples[key].append(values.mean(axis=(1, 2)))
    output: dict[str, dict[str, np.ndarray]] = {}
    for (metric, effect), chunks in samples.items():
        output.setdefault(metric, {})[effect] = np.concatenate(chunks)
    return output


def point_metrics(
    cells: dict[str, list[list[np.ndarray]]],
) -> dict[str, list[list[dict[str, float]]]]:
    validate_statistics(cells)
    return {
        cell: [
            [metrics_from_sufficient(matrix.sum(axis=0)) for matrix in seed]
            for seed in cells[cell]
        ]
        for cell in CELLS
    }


def factorial_point_summary(
    cells: dict[str, list[list[np.ndarray]]],
    samples: dict[str, dict[str, np.ndarray]],
    seeds: list[int],
    datasets: list[str],
) -> dict[str, dict[str, object]]:
    cell_metrics = point_metrics(cells)
    seed_count, dataset_count = validate_statistics(cells)
    if len(seeds) != seed_count or len(datasets) != dataset_count:
        raise ValueError("Seed or dataset labels do not match the factorial statistics")
    metric_names = sorted(cell_metrics["base"][0][0])
    output: dict[str, dict[str, object]] = {}
    for metric in metric_names:
        cell_values = {
            cell: np.asarray(
                [
                    [cell_metrics[cell][i][j][metric] for j in range(dataset_count)]
                    for i in range(seed_count)
                ],
                dtype=np.float64,
            )
            for cell in CELLS
        }
        effect_grid = {
            effect: np.empty((seed_count, dataset_count), dtype=np.float64)
            for effect in EFFECTS
        }
        for seed_index in range(seed_count):
            for dataset_index in range(dataset_count):
                point_cells = {
                    cell: {metric: cell_values[cell][seed_index, dataset_index]}
                    for cell in CELLS
                }
                effects = factorial_effects(point_cells)[metric]
                for effect, value in effects.items():
                    effect_grid[effect][seed_index, dataset_index] = value
        metric_output: dict[str, object] = {
            "cell_equal_seed_dataset_means": {
                cell: float(values.mean()) for cell, values in cell_values.items()
            }
        }
        for effect in EFFECTS:
            distribution = samples[metric][effect]
            grid = effect_grid[effect]
            metric_output[effect] = {
                "estimate": float(grid.mean()),
                "ci95_low": float(np.percentile(distribution, 2.5)),
                "ci95_high": float(np.percentile(distribution, 97.5)),
                "probability_greater_than_zero": float(np.mean(distribution > 0.0)),
                "per_seed_equal_dataset_effect": {
                    str(seed): float(value)
                    for seed, value in zip(seeds, grid.mean(axis=1), strict=True)
                },
                "per_dataset_equal_seed_effect": {
                    dataset: float(value)
                    for dataset, value in zip(datasets, grid.mean(axis=0), strict=True)
                },
            }
        output[metric] = metric_output
    return output


def main() -> None:
    args = parse_args()
    if not args.seeds or not args.relative_dirs:
        raise ValueError("Seeds and relative directories must be non-zero")
    stems = {cell: getattr(args, f"{cell}_stem") for cell in CELLS}
    cells: dict[str, list[list[np.ndarray]]] = {cell: [] for cell in CELLS}
    groups: dict[str, dict[str, int]] = {}
    for seed in args.seeds:
        groups[str(seed)] = {}
        for cell in CELLS:
            cells[cell].append([])
        for relative_dir in args.relative_dirs:
            expected_groups: list[str] | None = None
            for cell in CELLS:
                group_ids, statistics = load_seed_statistics(
                    args.outputs_root,
                    stems[cell],
                    seed,
                    relative_dir,
                    args.task_file,
                    args.instance_file,
                    args.cluster_key,
                )
                if expected_groups is None:
                    expected_groups = group_ids
                elif group_ids != expected_groups:
                    raise ValueError(
                        f"Factorial groups differ for seed {seed}, {relative_dir}"
                    )
                cells[cell][-1].append(statistics)
            assert expected_groups is not None
            groups[str(seed)][relative_dir] = len(expected_groups)

    samples = factorial_hierarchical_samples(
        cells,
        args.replicates,
        np.random.default_rng(args.seed),
    )
    output = {
        "bootstrap_unit": f"seed_then_dataset_then_{args.cluster_key}",
        "cells": stems,
        "dataset_weighting": "equal",
        "difference": "dynamic_gram_by_gsc_factorial",
        "groups": groups,
        "metrics": factorial_point_summary(
            cells,
            samples,
            args.seeds,
            args.relative_dirs,
        ),
        "relative_dirs": args.relative_dirs,
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
