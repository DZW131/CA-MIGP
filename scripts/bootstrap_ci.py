from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STAT_COLUMNS = (
    "detection_true_positive",
    "detection_false_positive",
    "detection_false_negative",
    "segmentation_dice_sum",
    "segmentation_count",
    "instance_true_positive",
    "instance_false_positive",
    "instance_false_negative",
    "matched_iou_sum",
    "aji_plus_sum",
    "pq_sum",
    "instance_count",
    "boundary_f1_sum",
    "count_absolute_error_sum",
    "diagnostic_count",
    "contact_instances",
    "contact_instances_detected",
    "contact_pairs",
    "contact_pair_successes",
    "contact_pair_merges",
    "segmentation_iou_sum",
)
STAT_INDEX = {name: index for index, name in enumerate(STAT_COLUMNS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap task and instance metrics")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cluster-key", choices=("domain", "sample_id"), default="sample_id")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def point_metrics(
    task_records: list[dict[str, float | str]],
    instance_records: list[dict[str, float | str]],
) -> dict[str, float]:
    values = metrics_from_sufficient(
        sufficient_statistics(task_records, instance_records)
    )
    return {key: float(np.asarray(value)) for key, value in values.items()}


def sufficient_statistics(
    task_records: list[dict[str, float | str]],
    instance_records: list[dict[str, float | str]],
) -> np.ndarray:
    """Reduce records to the additive terms required by all reported metrics."""
    stats = np.zeros(len(STAT_COLUMNS), dtype=np.float64)
    for record in task_records:
        if record["operation"] == "detection":
            stats[STAT_INDEX["detection_true_positive"]] += float(
                record["true_positive"]
            )
            stats[STAT_INDEX["detection_false_positive"]] += float(
                record["false_positive"]
            )
            stats[STAT_INDEX["detection_false_negative"]] += float(
                record["false_negative"]
            )
        elif record["operation"] == "segmentation":
            stats[STAT_INDEX["segmentation_dice_sum"]] += float(record["dice"])
            stats[STAT_INDEX["segmentation_iou_sum"]] += float(record["iou"])
            stats[STAT_INDEX["segmentation_count"]] += 1.0
    for record in instance_records:
        true_positive = float(record["true_positive"])
        stats[STAT_INDEX["instance_true_positive"]] += true_positive
        stats[STAT_INDEX["instance_false_positive"]] += float(
            record["false_positive"]
        )
        stats[STAT_INDEX["instance_false_negative"]] += float(
            record["false_negative"]
        )
        stats[STAT_INDEX["matched_iou_sum"]] += true_positive * float(record["sq"])
        stats[STAT_INDEX["aji_plus_sum"]] += float(record["aji_plus"])
        stats[STAT_INDEX["pq_sum"]] += float(record["pq"])
        stats[STAT_INDEX["instance_count"]] += 1.0
        if "boundary_f1" in record:
            stats[STAT_INDEX["boundary_f1_sum"]] += float(record["boundary_f1"])
            stats[STAT_INDEX["count_absolute_error_sum"]] += float(
                record["count_absolute_error"]
            )
            stats[STAT_INDEX["diagnostic_count"]] += 1.0
            for name in (
                "contact_instances",
                "contact_instances_detected",
                "contact_pairs",
                "contact_pair_successes",
                "contact_pair_merges",
            ):
                stats[STAT_INDEX[name]] += float(record[name])
    return stats


def metrics_from_sufficient(stats: np.ndarray) -> dict[str, np.ndarray]:
    """Convert one or many rows of additive statistics into metric values."""
    stats = np.asarray(stats, dtype=np.float64)

    def get(name: str) -> np.ndarray:
        return stats[..., STAT_INDEX[name]]

    detection_tp = get("detection_true_positive")
    precision = detection_tp / np.maximum(
        detection_tp + get("detection_false_positive"), 1.0
    )
    recall = detection_tp / np.maximum(
        detection_tp + get("detection_false_negative"), 1.0
    )
    instance_tp = get("instance_true_positive")
    dq = instance_tp / np.maximum(
        instance_tp
        + 0.5 * get("instance_false_positive")
        + 0.5 * get("instance_false_negative"),
        1.0,
    )
    sq = get("matched_iou_sum") / np.maximum(instance_tp, 1.0)
    metrics = {
        "detection_f1": 2.0 * precision * recall / np.maximum(
            precision + recall, 1e-12
        ),
        "segmentation_dice": get("segmentation_dice_sum")
        / np.maximum(get("segmentation_count"), 1.0),
        "segmentation_iou": get("segmentation_iou_sum")
        / np.maximum(get("segmentation_count"), 1.0),
        "mean_aji_plus": get("aji_plus_sum")
        / np.maximum(get("instance_count"), 1.0),
        "mean_pq": get("pq_sum") / np.maximum(get("instance_count"), 1.0),
        "global_dq": dq,
        "global_sq": sq,
        "global_pq": dq * sq,
    }
    if np.any(get("diagnostic_count") > 0):
        metrics.update(
            {
                "mean_boundary_f1": get("boundary_f1_sum")
                / np.maximum(get("diagnostic_count"), 1.0),
                "mean_count_absolute_error": get("count_absolute_error_sum")
                / np.maximum(get("diagnostic_count"), 1.0),
                "contact_instance_recall": get("contact_instances_detected")
                / np.maximum(get("contact_instances"), 1.0),
                "contact_pair_success_rate": get("contact_pair_successes")
                / np.maximum(get("contact_pairs"), 1.0),
                "contact_pair_merge_rate": get("contact_pair_merges")
                / np.maximum(get("contact_pairs"), 1.0),
            }
        )
    return metrics


def grouped_stat_matrix(
    task_groups: dict[str, list[dict[str, float | str]]],
    instance_groups: dict[str, list[dict[str, float | str]]],
    group_ids: list[str],
) -> np.ndarray:
    return np.stack(
        [
            sufficient_statistics(task_groups[group], instance_groups[group])
            for group in group_ids
        ]
    )


def bootstrap_metric_samples(
    stats: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
    *,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Vectorized cluster bootstrap from one sufficient-statistics row per group."""
    groups = stats.shape[0]
    probability = np.full(groups, 1.0 / groups)
    samples: dict[str, list[np.ndarray]] = {}
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        counts = rng.multinomial(groups, probability, size=size)
        metrics = metrics_from_sufficient(counts @ stats)
        for key, values in metrics.items():
            samples.setdefault(key, []).append(np.asarray(values))
    return {key: np.concatenate(values) for key, values in samples.items()}


def paired_bootstrap_metric_samples(
    candidate_stats: np.ndarray,
    reference_stats: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
    *,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Vectorized paired bootstrap using identical sampled group counts."""
    if candidate_stats.shape != reference_stats.shape:
        raise ValueError("Candidate and reference statistic matrices must match")
    groups = candidate_stats.shape[0]
    probability = np.full(groups, 1.0 / groups)
    samples: dict[str, list[np.ndarray]] = {}
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        counts = rng.multinomial(groups, probability, size=size)
        candidate = metrics_from_sufficient(counts @ candidate_stats)
        reference = metrics_from_sufficient(counts @ reference_stats)
        for key in candidate:
            samples.setdefault(key, []).append(candidate[key] - reference[key])
    return {key: np.concatenate(values) for key, values in samples.items()}


def grouped_records(
    records: list[dict[str, float | str]], key: str
) -> dict[str, list[dict[str, float | str]]]:
    groups: defaultdict[str, list[dict[str, float | str]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return dict(groups)


def propagate_group_key(
    records: list[dict[str, float | str]],
    companion: list[dict[str, float | str]],
    key: str,
) -> None:
    """Fill a legacy missing group key from companion rows sharing sample IDs."""
    if all(key in record for record in records):
        return
    mapping: dict[str, str] = {}
    for record in companion:
        if key not in record:
            continue
        sample_id = str(record["sample_id"])
        value = str(record[key])
        if sample_id in mapping and mapping[sample_id] != value:
            raise ValueError(f"Ambiguous {key} for sample {sample_id}")
        mapping[sample_id] = value
    for record in records:
        if key in record:
            continue
        sample_id = str(record["sample_id"])
        if sample_id not in mapping:
            raise ValueError(f"Cannot recover {key} for sample {sample_id}")
        record[key] = mapping[sample_id]


def main() -> None:
    args = parse_args()
    task_records = load_jsonl(args.tasks)
    instance_records = load_jsonl(args.instances)
    propagate_group_key(task_records, instance_records, args.cluster_key)
    propagate_group_key(instance_records, task_records, args.cluster_key)
    task_groups = grouped_records(task_records, args.cluster_key)
    instance_groups = grouped_records(instance_records, args.cluster_key)
    group_ids = sorted(set(task_groups) & set(instance_groups))
    if not group_ids:
        raise ValueError("No shared bootstrap groups between task and instance records")
    stats = grouped_stat_matrix(task_groups, instance_groups, group_ids)
    rng = np.random.default_rng(args.seed)
    samples = bootstrap_metric_samples(stats, args.replicates, rng)

    estimates = point_metrics(task_records, instance_records)
    output = {
        "bootstrap_unit": args.cluster_key,
        "groups": len(group_ids),
        "replicates": args.replicates,
        "seed": args.seed,
        "metrics": {
            key: {
                "estimate": value,
                "ci95_low": float(np.percentile(samples[key], 2.5)),
                "ci95_high": float(np.percentile(samples[key], 97.5)),
            }
            for key, value in estimates.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
