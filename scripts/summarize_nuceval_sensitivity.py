from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

FAMILIES = {
    "full": "pannuke_full_prompt",
    "camigp": "pannuke_geometry_no_orth",
}
METRICS = ("dice", "aji", "dq", "sq", "pq")
AGGREGATIONS = ("unweighted", "nucleus_count_weighted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize NucEval sensitivity runs")
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def collect(outputs_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for family, stem in FAMILIES.items():
        for seed in (17, 23, 42):
            path = (
                outputs_root
                / f"{stem}_seed{seed}"
                / "nuinsseg_nuceval"
                / "ood_test_nuceval_sensitivity_summary.json"
            )
            payload = read_json(path)
            if payload.get("nuceval_sha256") != (
                "54ddb315af7547a7f465ae2d1208e16949669c913a164e3a80e59c9898180788"
            ):
                raise ValueError(f"Unexpected NucEval provenance: {path}")
            results = payload.get("results")
            if not isinstance(results, dict) or set(results) != {"0", "1"}:
                raise ValueError(f"Expected NucEval zones 0 and 1: {path}")
            for zone_width, zone in results.items():
                if not isinstance(zone, dict) or int(zone.get("images", 0)) != 665:
                    raise ValueError(f"Incomplete NucEval zone: {path}:{zone_width}")
                for aggregation in AGGREGATIONS:
                    values = zone.get(aggregation)
                    if not isinstance(values, dict):
                        raise ValueError(
                            f"Missing NucEval aggregation: {path}:{zone_width}:{aggregation}"
                        )
                    for metric in METRICS:
                        value = float(values[metric])
                        if not 0.0 <= value <= 1.0:
                            raise ValueError(f"Invalid NucEval metric: {path}:{metric}={value}")
                        records.append(
                            {
                                "family": family,
                                "seed": seed,
                                "zone_width": int(zone_width),
                                "aggregation": aggregation,
                                "metric": metric,
                                "value": value,
                            }
                        )
    return records


def aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str], list[tuple[int, float]]] = {}
    for record in records:
        key = (
            str(record["family"]),
            int(record["zone_width"]),
            str(record["aggregation"]),
            str(record["metric"]),
        )
        grouped.setdefault(key, []).append((int(record["seed"]), float(record["value"])))
    output: list[dict[str, object]] = []
    for (family, zone_width, aggregation, metric), items in sorted(grouped.items()):
        items.sort()
        values = [value for _, value in items]
        if len(values) != 3:
            raise ValueError(
                f"Expected three seeds for {(family, zone_width, aggregation, metric)}"
            )
        output.append(
            {
                "family": family,
                "zone_width": zone_width,
                "aggregation": aggregation,
                "metric": metric,
                "seeds": 3,
                "mean": mean(values),
                "sample_sd": stdev(values),
                **{f"seed_{seed}": value for seed, value in items},
            }
        )
    return output


def paired_differences(records: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (
            str(record["family"]),
            int(record["seed"]),
            int(record["zone_width"]),
            str(record["aggregation"]),
            str(record["metric"]),
        ): float(record["value"])
        for record in records
    }
    output: list[dict[str, object]] = []
    for zone_width in (0, 1):
        for aggregation in AGGREGATIONS:
            for metric in METRICS:
                differences = [
                    indexed[("camigp", seed, zone_width, aggregation, metric)]
                    - indexed[("full", seed, zone_width, aggregation, metric)]
                    for seed in (17, 23, 42)
                ]
                output.append(
                    {
                        "comparison": "camigp_minus_full",
                        "zone_width": zone_width,
                        "aggregation": aggregation,
                        "metric": metric,
                        "seeds": 3,
                        "mean_difference": mean(differences),
                        "sample_sd": stdev(differences),
                        "positive_seeds": sum(value > 0 for value in differences),
                        **{
                            f"seed_{seed}": value
                            for seed, value in zip((17, 23, 42), differences, strict=True)
                        },
                    }
                )
    return output


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.with_suffix(".json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = collect(args.outputs_root)
    aggregates = aggregate(records)
    differences = paired_differences(records)
    write_records(args.output_dir / "nuceval_scores", records)
    write_records(args.output_dir / "nuceval_aggregate", aggregates)
    write_records(args.output_dir / "nuceval_differences", differences)
    print(
        json.dumps(
            {
                "score_rows": len(records),
                "aggregate_rows": len(aggregates),
                "difference_rows": len(differences),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
