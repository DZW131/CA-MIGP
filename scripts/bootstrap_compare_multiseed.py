from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bootstrap_ci import (
    grouped_records,
    grouped_stat_matrix,
    load_jsonl,
    metrics_from_sufficient,
    propagate_group_key,
)
from bootstrap_compare import LOWER_IS_BETTER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-level paired bootstrap across seeds and dataset groups"
    )
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--candidate-stem", required=True)
    parser.add_argument("--reference-stem", required=True)
    parser.add_argument("--relative-dir", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--instance-file", required=True)
    parser.add_argument("--cluster-key", choices=("domain", "sample_id"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23, 42))
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_seed_statistics(
    outputs_root: Path,
    stem: str,
    seed: int,
    relative_dir: str,
    task_file: str,
    instance_file: str,
    cluster_key: str,
) -> tuple[list[str], np.ndarray]:
    evaluation = outputs_root / f"{stem}_seed{seed}" / relative_dir
    tasks = load_jsonl(evaluation / task_file)
    instances = load_jsonl(evaluation / instance_file)
    propagate_group_key(tasks, instances, cluster_key)
    propagate_group_key(instances, tasks, cluster_key)
    task_groups = grouped_records(tasks, cluster_key)
    instance_groups = grouped_records(instances, cluster_key)
    group_ids = sorted(set(task_groups) & set(instance_groups))
    if not group_ids:
        raise ValueError(f"No shared groups for {stem} seed {seed}: {evaluation}")
    if set(task_groups) != set(group_ids) or set(instance_groups) != set(group_ids):
        raise ValueError(f"Task/instance group mismatch for {stem} seed {seed}")
    return group_ids, grouped_stat_matrix(task_groups, instance_groups, group_ids)


def hierarchical_paired_samples(
    candidate: list[np.ndarray],
    reference: list[np.ndarray],
    replicates: int,
    rng: np.random.Generator,
    *,
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("Candidate and reference must contain the same non-zero seeds")
    if any(
        left.shape != right.shape
        for left, right in zip(candidate, reference, strict=True)
    ):
        raise ValueError("Candidate and reference group matrices must be paired by seed")
    seed_count = len(candidate)
    samples: dict[str, list[np.ndarray]] = {}
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        chosen_seeds = rng.integers(0, seed_count, size=(size, seed_count))
        chunk_metrics: dict[str, np.ndarray] = {}
        for position in range(seed_count):
            position_seed = chosen_seeds[:, position]
            for seed_index, (candidate_stats, reference_stats) in enumerate(
                zip(candidate, reference, strict=True)
            ):
                selected = np.flatnonzero(position_seed == seed_index)
                if not selected.size:
                    continue
                groups = candidate_stats.shape[0]
                counts = rng.multinomial(
                    groups,
                    np.full(groups, 1.0 / groups),
                    size=selected.size,
                )
                candidate_metrics = metrics_from_sufficient(counts @ candidate_stats)
                reference_metrics = metrics_from_sufficient(counts @ reference_stats)
                for key in candidate_metrics:
                    values = chunk_metrics.setdefault(
                        key, np.zeros((size, seed_count), dtype=np.float64)
                    )
                    values[selected, position] = (
                        candidate_metrics[key] - reference_metrics[key]
                    )
        for key, values in chunk_metrics.items():
            samples.setdefault(key, []).append(values.mean(axis=1))
    return {key: np.concatenate(values) for key, values in samples.items()}


def summarize(
    candidate: list[np.ndarray],
    reference: list[np.ndarray],
    samples: dict[str, np.ndarray],
    seeds: list[int],
) -> dict[str, dict[str, object]]:
    candidate_metrics = [
        metrics_from_sufficient(stats.sum(axis=0)) for stats in candidate
    ]
    reference_metrics = [
        metrics_from_sufficient(stats.sum(axis=0)) for stats in reference
    ]
    output: dict[str, dict[str, object]] = {}
    for key, values in samples.items():
        candidate_values = np.asarray(
            [float(metrics[key]) for metrics in candidate_metrics]
        )
        reference_values = np.asarray(
            [float(metrics[key]) for metrics in reference_metrics]
        )
        differences = candidate_values - reference_values
        lower_is_better = key in LOWER_IS_BETTER
        output[key] = {
            "candidate_mean": float(candidate_values.mean()),
            "reference_mean": float(reference_values.mean()),
            "estimate": float(differences.mean()),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "probability_candidate_better": float(
                np.mean(values < 0.0) if lower_is_better else np.mean(values > 0.0)
            ),
            "per_seed_difference": {
                str(seed): float(value)
                for seed, value in zip(seeds, differences, strict=True)
            },
        }
    return output


def main() -> None:
    args = parse_args()
    if args.replicates <= 0 or not args.seeds:
        raise ValueError("Replicates and seed list must be non-zero")
    candidate: list[np.ndarray] = []
    reference: list[np.ndarray] = []
    groups_by_seed: dict[str, int] = {}
    for seed in args.seeds:
        candidate_groups, candidate_stats = load_seed_statistics(
            args.outputs_root,
            args.candidate_stem,
            seed,
            args.relative_dir,
            args.task_file,
            args.instance_file,
            args.cluster_key,
        )
        reference_groups, reference_stats = load_seed_statistics(
            args.outputs_root,
            args.reference_stem,
            seed,
            args.relative_dir,
            args.task_file,
            args.instance_file,
            args.cluster_key,
        )
        if candidate_groups != reference_groups:
            raise ValueError(f"Candidate/reference groups differ for seed {seed}")
        candidate.append(candidate_stats)
        reference.append(reference_stats)
        groups_by_seed[str(seed)] = len(candidate_groups)

    rng = np.random.default_rng(args.seed)
    samples = hierarchical_paired_samples(
        candidate, reference, args.replicates, rng
    )
    output = {
        "bootstrap_unit": f"seed_then_{args.cluster_key}",
        "candidate": args.candidate_stem,
        "reference": args.reference_stem,
        "difference": "candidate_minus_reference",
        "seeds": args.seeds,
        "groups_by_seed": groups_by_seed,
        "replicates": args.replicates,
        "seed": args.seed,
        "metrics": summarize(candidate, reference, samples, args.seeds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
