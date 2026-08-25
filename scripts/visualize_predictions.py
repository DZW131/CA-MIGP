from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data import ManifestTaskDataset  # noqa: E402
from pathm.metrics import heatmap_points, instance_scores  # noqa: E402
from pathm.models import build_model  # noqa: E402
from pathm.postprocessing import seeded_watershed_instances  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render qualitative prediction panels")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("val", "test", "ood_test"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--segmentation-threshold", type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def overlay_boundary(image: np.ndarray, labels: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    rendered = image.copy()
    rendered[find_boundaries(labels, mode="inner")] = color
    return Image.fromarray(rendered)


def error_overlay(
    image: np.ndarray, reference: np.ndarray, prediction: np.ndarray
) -> Image.Image:
    reference_foreground = reference > 0
    prediction_foreground = prediction > 0
    rendered = image.astype(np.float32) * 0.55
    rendered[reference_foreground & prediction_foreground] += np.array([30, 145, 30])
    rendered[~reference_foreground & prediction_foreground] += np.array([150, 20, 20])
    rendered[reference_foreground & ~prediction_foreground] += np.array([20, 70, 165])
    return Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8))


def add_label(panel: Image.Image, label: str) -> Image.Image:
    header = 24
    rendered = Image.new("RGB", (panel.width, panel.height + header), "white")
    rendered.paste(panel, (0, header))
    ImageDraw.Draw(rendered).text((6, 5), label, fill="black", font=ImageFont.load_default())
    return rendered


def main() -> None:
    args = parse_args()
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.manifest is not None:
        config["data"]["manifest"] = str(args.manifest.resolve())
    data_config = config["data"]
    split_name = data_config.get(f"{args.split}_split", args.split)
    dataset = ManifestTaskDataset(
        data_config["manifest"], split_name, operations=["segmentation"]
    )
    requested = set(args.sample_id)
    selected_indices: list[int] = []
    selected_domains: set[str] = set()
    for index, record in enumerate(dataset.records):
        if requested and record.patient_id not in requested:
            continue
        if not requested and record.domain in selected_domains:
            continue
        selected_indices.append(index)
        selected_domains.add(record.domain)
        if len(selected_indices) >= args.max_images:
            break
    if requested - {dataset.records[index].patient_id for index in selected_indices}:
        missing = sorted(requested - {dataset.records[index].patient_id for index in selected_indices})
        raise ValueError(f"Requested sample IDs not found: {missing}")
    if not selected_indices:
        raise ValueError("No samples selected")

    device = torch.device(args.device)
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    evaluation = config["evaluation"]
    detection_threshold = (
        args.detection_threshold
        if args.detection_threshold is not None
        else float(evaluation["detection_threshold"])
    )
    segmentation_threshold = (
        args.segmentation_threshold
        if args.segmentation_threshold is not None
        else float(evaluation["segmentation_threshold"])
    )
    output_records: list[dict[str, float | str]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for index in selected_indices:
            sample = dataset[index]
            image_tensor = sample["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            target_id = sample["target_id"].reshape(1).to(device)  # type: ignore[union-attr]
            detection_id = torch.zeros(1, dtype=torch.long, device=device)
            segmentation_id = torch.ones(1, dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bool(config["training"].get("amp", True)) and device.type == "cuda",
            ):
                detection_logits = model(image_tensor, detection_id, target_id)["logits"]
                segmentation_logits = model(image_tensor, segmentation_id, target_id)["logits"]
            detection_map = torch.sigmoid(detection_logits.float())[0, 0].cpu().numpy()
            segmentation_map = torch.sigmoid(segmentation_logits.float())[0, 0].cpu().numpy()
            prediction = seeded_watershed_instances(
                segmentation_map,
                detection_map,
                segmentation_threshold=segmentation_threshold,
                detection_threshold=detection_threshold,
                detection_minimum_distance=int(evaluation["detection_minimum_distance"]),
            )
            reference = np.load(str(sample["label_path"]), allow_pickle=False)
            ignore_path = str(sample["ignore_path"])
            if ignore_path:
                ignore_mask = np.load(ignore_path, allow_pickle=False).astype(bool)
                prediction[ignore_mask] = 0
                reference[ignore_mask] = 0
            original = np.asarray(Image.open(str(sample["image_path"])).convert("RGB")).copy()
            prediction_panel = overlay_boundary(original, prediction, (0, 255, 255))
            draw = ImageDraw.Draw(prediction_panel)
            for row, column in heatmap_points(
                torch.from_numpy(detection_map),
                threshold=detection_threshold,
                minimum_distance=int(evaluation["detection_minimum_distance"]),
            ):
                draw.ellipse((column - 2, row - 2, column + 2, row + 2), fill=(255, 255, 0))
            panels = [
                add_label(Image.fromarray(original), "Image"),
                add_label(overlay_boundary(original, reference, (0, 255, 0)), "Reference"),
                add_label(prediction_panel, "Prediction"),
                add_label(error_overlay(original, reference, prediction), "TP / FP / FN"),
            ]
            canvas = Image.new(
                "RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels))
            )
            left = 0
            for panel in panels:
                canvas.paste(panel, (left, 0))
                left += panel.width
            sample_id = str(sample["sample_id"])
            output_path = args.output_dir / f"{sample_id.replace('/', '_')}.png"
            canvas.save(output_path)
            scores = instance_scores(reference, prediction)
            output_records.append(
                {
                    "sample_id": sample_id,
                    "domain": str(sample["domain"]),
                    "output": str(output_path),
                    **scores,
                }
            )

    (args.output_dir / "index.json").write_text(
        json.dumps(output_records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output_records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
