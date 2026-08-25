from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from pathm.models import build_model  # noqa: E402
from pathm.training import MixedTaskLoss, evaluate  # noqa: E402
from train import make_loader, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune one threshold on validation data")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", choices=("detection", "segmentation"), required=True)
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["data"]["operations"] = [args.task]
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        config["training"]["batch_size"] = args.batch_size
    seed = int(config["training"]["seed"])
    set_seed(seed, deterministic=True)
    loader = make_loader(config, "val", augment=False, seed=seed)
    device = torch.device(args.device)
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    loss_function = MixedTaskLoss()
    results: list[dict[str, float]] = []
    evaluation_config = config["evaluation"]

    for value in args.values:
        metrics = evaluate(
            model,
            loader,
            loss_function,
            device,
            use_amp=bool(config["training"].get("amp", True)),
            segmentation_threshold=value
            if args.task == "segmentation"
            else float(evaluation_config["segmentation_threshold"]),
            detection_threshold=value
            if args.task == "detection"
            else float(evaluation_config["detection_threshold"]),
            detection_minimum_distance=int(evaluation_config["detection_minimum_distance"]),
            detection_matching_radius=float(evaluation_config["detection_matching_radius"]),
        )
        metric_name = "detection_f1" if args.task == "detection" else "segmentation_dice"
        results.append({"threshold": value, metric_name: metrics[metric_name]})
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    metric_name = "detection_f1" if args.task == "detection" else "segmentation_dice"
    best = max(results, key=lambda result: result[metric_name])
    output = {
        "selection_split": "val",
        "task": args.task,
        "metric": metric_name,
        "selected_threshold": best["threshold"],
        "selected_metric": best[metric_name],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
