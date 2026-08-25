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
    paired_bootstrap_metric_samples,
    propagate_group_key,
)


LOWER_IS_BETTER = {"mean_count_absolute_error", "contact_pair_merge_rate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison of two models")
    parser.add_argument("--candidate-tasks", required=True, type=Path)
    parser.add_argument("--candidate-instances", required=True, type=Path)
    parser.add_argument("--reference-tasks", required=True, type=Path)
    parser.add_argument("--reference-instances", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cluster-key", choices=("domain", "sample_id"), default="sample_id")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_tasks = load_jsonl(args.candidate_tasks)
    candidate_instances = load_jsonl(args.candidate_instances)
    reference_tasks = load_jsonl(args.reference_tasks)
    reference_instances = load_jsonl(args.reference_instances)
    for tasks, instances in (
        (candidate_tasks, candidate_instances),
        (reference_tasks, reference_instances),
    ):
        propagate_group_key(tasks, instances, args.cluster_key)
        propagate_group_key(instances, tasks, args.cluster_key)
    grouped = {
        "candidate_tasks": grouped_records(candidate_tasks, args.cluster_key),
        "candidate_instances": grouped_records(candidate_instances, args.cluster_key),
        "reference_tasks": grouped_records(reference_tasks, args.cluster_key),
        "reference_instances": grouped_records(reference_instances, args.cluster_key),
    }
    group_ids = sorted(set.intersection(*(set(records) for records in grouped.values())))
    if not group_ids:
        raise ValueError("No shared bootstrap groups across candidate and reference records")
    for name, records in grouped.items():
        missing = set(records) - set(group_ids)
        if missing:
            raise ValueError(f"{name} has {len(missing)} groups absent from another input")

    candidate_stats = grouped_stat_matrix(
        grouped["candidate_tasks"], grouped["candidate_instances"], group_ids
    )
    reference_stats = grouped_stat_matrix(
        grouped["reference_tasks"], grouped["reference_instances"], group_ids
    )
    candidate_estimates = {
        key: float(value)
        for key, value in metrics_from_sufficient(candidate_stats.sum(axis=0)).items()
    }
    reference_estimates = {
        key: float(value)
        for key, value in metrics_from_sufficient(reference_stats.sum(axis=0)).items()
    }
    rng = np.random.default_rng(args.seed)
    differences = paired_bootstrap_metric_samples(
        candidate_stats, reference_stats, args.replicates, rng
    )

    output = {
        "bootstrap_unit": args.cluster_key,
        "groups": len(group_ids),
        "replicates": args.replicates,
        "seed": args.seed,
        "difference": "candidate_minus_reference",
        "metrics": {
            key: {
                "candidate": candidate_estimates[key],
                "reference": reference_estimates[key],
                "estimate": candidate_estimates[key] - reference_estimates[key],
                "ci95_low": float(np.percentile(differences[key], 2.5)),
                "ci95_high": float(np.percentile(differences[key], 97.5)),
                "probability_greater_than_zero": float(
                    np.mean(np.asarray(differences[key]) > 0.0)
                ),
                "direction": "lower_is_better"
                if key in LOWER_IS_BETTER
                else "higher_is_better",
                "probability_candidate_better": float(
                    np.mean(np.asarray(differences[key]) < 0.0)
                    if key in LOWER_IS_BETTER
                    else np.mean(np.asarray(differences[key]) > 0.0)
                ),
            }
            for key in candidate_estimates
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
