from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate a unified model checkpoint")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stain-h-scale", type=float, default=1.0)
    parser.add_argument("--stain-e-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", choices=("val", "test", "ood_test"), default="val")
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--segmentation-threshold", type=float)
    parser.add_argument("--per-image-output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--mismatch-style", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.manifest is not None:
        config["data"]["manifest"] = str(args.manifest.resolve())
    config["data"]["stain_h_scale"] = args.stain_h_scale
    config["data"]["stain_e_scale"] = args.stain_e_scale
    training = config["training"]
    evaluation_config = config["evaluation"]
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        training["batch_size"] = args.batch_size
    if args.split in {"test", "ood_test"}:
        config["data"][f"{args.split}_split"] = args.split
    seed = int(training["seed"])
    set_seed(seed, deterministic=True)
    device = torch.device(args.device)
    loader = make_loader(
        config,
        args.split,
        augment=False,
        seed=seed,
        mismatched_style=args.mismatch_style,
    )
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    per_image_records: list[dict[str, float | str]] = []
    metrics = evaluate(
        model,
        loader,
        MixedTaskLoss(),
        device,
        use_amp=bool(training.get("amp", True)),
        segmentation_threshold=args.segmentation_threshold
        if args.segmentation_threshold is not None
        else float(evaluation_config["segmentation_threshold"]),
        detection_threshold=args.detection_threshold
        if args.detection_threshold is not None
        else float(evaluation_config["detection_threshold"]),
        detection_minimum_distance=int(evaluation_config["detection_minimum_distance"]),
        detection_matching_radius=float(evaluation_config["detection_matching_radius"]),
        per_image_records=per_image_records,
        mismatch_style=args.mismatch_style,
    )
    if args.mismatch_style:
        metrics["style_source_policy"] = "different_domain_else_patient_else_image"
    if args.per_image_output is not None:
        args.per_image_output.parent.mkdir(parents=True, exist_ok=True)
        args.per_image_output.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in per_image_records),
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
