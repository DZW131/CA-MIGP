"""Recompute paired five-seed factorial mean-PQ effects from published run summaries."""

import csv
import math
from pathlib import Path

from scipy.stats import t


HERE = Path(__file__).resolve().parent
SEEDS = (17, 23, 42, 101, 202)
DATASETS = ("frozen_test", "nuinsseg_ood", "cryonuseg_ann1_round1")
CONTRASTS = {
    "B-A": lambda v: v["B"] - v["A"],
    "C-A": lambda v: v["C"] - v["A"],
    "D-B": lambda v: v["D"] - v["B"],
    "D-C": lambda v: v["D"] - v["C"],
    "D-A": lambda v: v["D"] - v["A"],
    "interaction": lambda v: (v["D"] - v["B"]) - (v["C"] - v["A"]),
}


def summary(values):
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    half_width = t.ppf(0.975, df=len(values) - 1) * sd / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def main():
    data = {}
    with (HERE / "all_run_summary_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["cell"] in "ABCD" and row["dataset"] in DATASETS and int(row["seed"]) in SEEDS:
                key = (row["dataset"], int(row["seed"]), row["cell"])
                if key in data:
                    raise ValueError(f"Duplicate record: {key}")
                data[key] = float(row["mean_pq"])
    assert len(data) == 60, f"Expected 4 cells x 5 seeds x 3 datasets, found {len(data)}"

    for dataset in (*DATASETS, "equal_dataset"):
        print(dataset)
        for name, contrast in CONTRASTS.items():
            differences = []
            for seed in SEEDS:
                if dataset == "equal_dataset":
                    values = {
                        cell: sum(data[(cohort, seed, cell)] for cohort in DATASETS) / len(DATASETS)
                        for cell in "ABCD"
                    }
                else:
                    values = {cell: data[(dataset, seed, cell)] for cell in "ABCD"}
                differences.append(contrast(values))
            mean, low, high = summary(differences)
            print(f"  {name:12} {mean:+.4f} [{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
