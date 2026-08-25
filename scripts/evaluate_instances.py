from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import loadmat
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data import ManifestTaskDataset  # noqa: E402
from pathm.metrics import (  # noqa: E402
    instance_diagnostic_scores,
    instance_scores,
    nucleus_count_weighted_mean,
)
from pathm.models import build_model  # noqa: E402
from pathm.postprocessing import seeded_watershed_instances  # noqa: E402

NUCEVAL_COMMIT = "45e90c2b468da025bd304ef426754b995d6a90d3"
NUCEVAL_SHA256 = "54ddb315af7547a7f465ae2d1208e16949669c913a164e3a80e59c9898180788"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fused nucleus instances")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detection-checkpoint", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stain-h-scale", type=float, default=1.0)
    parser.add_argument("--stain-e-scale", type=float, default=1.0)
    parser.add_argument("--split", choices=("val", "test", "ood_test"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--detection-threshold", type=float)
    parser.add_argument("--segmentation-threshold", type=float)
    parser.add_argument("--mismatch-style", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--contact-radius", type=float, default=3.0)
    parser.add_argument("--boundary-tolerance", type=float, default=2.0)
    parser.add_argument("--nuceval-root", type=Path)
    parser.add_argument("--nuinsseg-raw-root", type=Path)
    parser.add_argument("--nuceval-zone-widths", nargs="+", type=int, default=(0, 1))
    return parser.parse_args()


def load_nuceval(root: Path) -> Callable[..., dict[str, float | int]]:
    source = root / "nuceval.py"
    if not source.is_file():
        raise ValueError(f"Missing pinned NucEval source: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != NUCEVAL_SHA256:
        raise ValueError(f"Unexpected NucEval source hash: {digest}")
    spec = importlib.util.spec_from_file_location("pinned_nuceval", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load NucEval source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "NucEval", None)
    if not callable(function):
        raise ValueError(f"NucEval function not found: {source}")
    return function


def nuinsseg_stacked_mask_path(raw_root: Path, domain: str, image_path: str) -> Path:
    prefix = "NuInsSeg:"
    if not domain.startswith(prefix):
        raise ValueError(f"NucEval sensitivity requires a NuInsSeg domain: {domain}")
    organ = domain[len(prefix) :]
    return raw_root / "extracted" / organ / "stacked mask" / f"{Path(image_path).stem}.mat"


def load_nuinsseg_stacked_masks(
    raw_root: Path,
    domain: str,
    image_path: str,
    expected_shape: tuple[int, int],
) -> tuple[list[np.ndarray], Path, int]:
    source = nuinsseg_stacked_mask_path(raw_root, domain, image_path)
    if not source.is_file():
        raise ValueError(f"Missing NuInsSeg stacked-mask file: {source}")
    payload = loadmat(source)
    keys = {key for key in payload if not key.startswith("__")}
    if keys != {"stacked_mask"}:
        raise ValueError(f"Unexpected NuInsSeg MAT keys in {source}: {sorted(keys)}")
    stack = np.asarray(payload["stacked_mask"])
    if stack.ndim == 2:
        stack = stack[:, :, None]
    if stack.ndim != 3 or tuple(stack.shape[:2]) != expected_shape:
        raise ValueError(
            f"Unexpected NuInsSeg stacked-mask shape in {source}: {stack.shape}; "
            f"expected {expected_shape} x instances"
        )
    if not np.all((stack == 0) | (stack == 1)):
        raise ValueError(f"NuInsSeg stacked masks must be binary: {source}")
    masks = [np.asarray(stack[:, :, index], dtype=np.uint8) for index in range(stack.shape[2])]
    nonempty_masks = [mask for mask in masks if np.any(mask)]
    empty_masks_removed = len(masks) - len(nonempty_masks)
    if not nonempty_masks:
        raise ValueError(f"NuInsSeg stacked-mask file has no non-empty instances: {source}")
    return nonempty_masks, source, empty_masks_removed


def summarize_nuceval_records(
    records: list[dict[str, object]],
) -> dict[str, object]:
    metrics = ("dice", "aji", "dq", "sq", "pq")
    output: dict[str, object] = {}
    for zone_width in sorted({int(record["zone_width"]) for record in records}):
        selected = [record for record in records if int(record["zone_width"]) == zone_width]
        if not selected:
            continue
        output[str(zone_width)] = {
            "images": len(selected),
            "reference_nuclei": int(
                sum(float(record["reference_instances"]) for record in selected)
            ),
            "unweighted": {
                metric: sum(float(record[metric]) for record in selected) / len(selected)
                for metric in metrics
            },
            "nucleus_count_weighted": {
                metric: nucleus_count_weighted_mean(selected, metric) for metric in metrics
            },
        }
    return output


def main() -> None:
    args = parse_args()
    if (args.nuceval_root is None) != (args.nuinsseg_raw_root is None):
        raise ValueError("Provide both --nuceval-root and --nuinsseg-raw-root")
    if any(width < 0 for width in args.nuceval_zone_widths):
        raise ValueError("NucEval zone widths must be non-negative")
    nuceval = load_nuceval(args.nuceval_root) if args.nuceval_root else None
    with args.config.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.manifest is not None:
        config["data"]["manifest"] = str(args.manifest.resolve())
    config["data"]["stain_h_scale"] = args.stain_h_scale
    config["data"]["stain_e_scale"] = args.stain_e_scale
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    split_name = data_config[f"{args.split}_split"] if args.split == "val" else args.split
    dataset = ManifestTaskDataset(
        data_config["manifest"],
        split_name,
        operations=["segmentation"],
        max_records=args.max_records,
        stain_h_scale=args.stain_h_scale,
        stain_e_scale=args.stain_e_scale,
        mismatched_style=args.mismatch_style,
    )
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device)
    model = build_model(config["model"]).to(device)
    if args.checkpoint is not None:
        if args.detection_checkpoint is not None or args.segmentation_checkpoint is not None:
            raise ValueError("Use either --checkpoint or both specialist checkpoint arguments")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        detection_model = segmentation_model = model
    else:
        if args.detection_checkpoint is None or args.segmentation_checkpoint is None:
            raise ValueError(
                "Provide --checkpoint or both --detection-checkpoint and --segmentation-checkpoint"
            )
        detection_model = model
        segmentation_model = build_model(config["model"]).to(device)
        detection_state = torch.load(
            args.detection_checkpoint, map_location=device, weights_only=False
        )
        segmentation_state = torch.load(
            args.segmentation_checkpoint, map_location=device, weights_only=False
        )
        detection_model.load_state_dict(detection_state["model"])
        segmentation_model.load_state_dict(segmentation_state["model"])
    detection_model.eval()
    segmentation_model.eval()
    detection_threshold = (
        args.detection_threshold
        if args.detection_threshold is not None
        else float(evaluation_config["detection_threshold"])
    )
    segmentation_threshold = (
        args.segmentation_threshold
        if args.segmentation_threshold is not None
        else float(evaluation_config["segmentation_threshold"])
    )
    records: list[dict[str, float | str]] = []
    nuceval_records: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target_ids = batch["target_id"].to(device, non_blocking=True)
            detection_ids = torch.zeros(images.shape[0], dtype=torch.long, device=device)
            segmentation_ids = torch.ones(images.shape[0], dtype=torch.long, device=device)
            style_images = (
                batch["style_image"].to(device, non_blocking=True) if args.mismatch_style else None
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bool(config["training"].get("amp", True)) and device.type == "cuda",
            ):
                if style_images is None:
                    detection_logits = detection_model(images, detection_ids, target_ids)["logits"]
                    segmentation_logits = segmentation_model(images, segmentation_ids, target_ids)[
                        "logits"
                    ]
                else:
                    detection_logits = detection_model(
                        images,
                        detection_ids,
                        target_ids,
                        style_images=style_images,
                    )["logits"]
                    segmentation_logits = segmentation_model(
                        images,
                        segmentation_ids,
                        target_ids,
                        style_images=style_images,
                    )["logits"]
            detection_maps = torch.sigmoid(detection_logits.float()).cpu().numpy()[:, 0]
            segmentation_maps = torch.sigmoid(segmentation_logits.float()).cpu().numpy()[:, 0]
            for index, sample_id in enumerate(batch["sample_id"]):
                prediction = seeded_watershed_instances(
                    segmentation_maps[index],
                    detection_maps[index],
                    segmentation_threshold=segmentation_threshold,
                    detection_threshold=detection_threshold,
                    detection_minimum_distance=int(evaluation_config["detection_minimum_distance"]),
                )
                reference = np.load(batch["label_path"][index], allow_pickle=False)
                ignore_path = batch["ignore_path"][index]
                ignore_mask = None
                if ignore_path:
                    ignore_mask = np.load(ignore_path, allow_pickle=False).astype(bool)
                if nuceval is not None:
                    ground_truth_masks, ground_truth_source, empty_masks_removed = (
                        load_nuinsseg_stacked_masks(
                            args.nuinsseg_raw_root,
                            batch["domain"][index],
                            batch["image_path"][index],
                            prediction.shape,
                        )
                    )
                    for zone_width in args.nuceval_zone_widths:
                        sensitivity = nuceval(
                            ground_truth_masks,
                            prediction.copy(),
                            amb=ignore_mask,
                            normalized=True,
                            zone_width=zone_width,
                            overlap_thresh_amb=0.25,
                            match_iou=0.5,
                        )
                        nuceval_records.append(
                            {
                                "sample_id": Path(batch["image_path"][index]).stem,
                                "domain": batch["domain"][index],
                                "zone_width": zone_width,
                                "ground_truth_source": str(ground_truth_source),
                                "source_instances": len(ground_truth_masks),
                                "empty_source_instances_removed": empty_masks_removed,
                                "reference_instances": int(sensitivity["num_nuclei"]),
                                **{
                                    metric: float(sensitivity[metric])
                                    for metric in ("dice", "aji", "dq", "sq", "pq")
                                },
                            }
                        )
                if ignore_mask is not None:
                    prediction[ignore_mask] = 0
                    reference[ignore_mask] = 0
                scores = instance_scores(reference, prediction)
                diagnostics = instance_diagnostic_scores(
                    reference,
                    prediction,
                    contact_radius=args.contact_radius,
                    boundary_tolerance=args.boundary_tolerance,
                )
                record = {
                    "sample_id": sample_id,
                    "domain": batch["domain"][index],
                    "image_path": batch["image_path"][index],
                    **scores,
                    **diagnostics,
                }
                if args.mismatch_style:
                    record.update(
                        {
                            "image_path": batch["image_path"][index],
                            "style_source_sample_id": batch["style_source_sample_id"][index],
                            "style_source_domain": batch["style_source_domain"][index],
                            "style_source_image": batch["style_source_image"][index],
                        }
                    )
                records.append(record)

    true_positive = sum(record["true_positive"] for record in records)
    false_positive = sum(record["false_positive"] for record in records)
    false_negative = sum(record["false_negative"] for record in records)
    matched_iou = sum(record["true_positive"] * record["sq"] for record in records)
    global_dq = true_positive / max(
        true_positive + 0.5 * false_positive + 0.5 * false_negative, 1.0
    )
    global_sq = matched_iou / max(true_positive, 1.0)
    contact_instances = sum(record["contact_instances"] for record in records)
    contact_pairs = sum(record["contact_pairs"] for record in records)
    summary = {
        "split": args.split,
        "images": len(records),
        "detection_threshold": detection_threshold,
        "segmentation_threshold": segmentation_threshold,
        "mean_aji_plus": sum(record["aji_plus"] for record in records) / len(records),
        "mean_pq": sum(record["pq"] for record in records) / len(records),
        "nucleus_count_weighted_pq": nucleus_count_weighted_mean(records, "pq"),
        "global_dq": global_dq,
        "global_sq": global_sq,
        "global_pq": global_dq * global_sq,
        "mean_boundary_f1": sum(record["boundary_f1"] for record in records) / len(records),
        "mean_count_absolute_error": sum(record["count_absolute_error"] for record in records)
        / len(records),
        "global_count_absolute_error": abs(
            sum(record["predicted_instances"] for record in records)
            - sum(record["reference_instances"] for record in records)
        ),
        "images_with_contacts": sum(record["contact_pairs"] > 0 for record in records),
        "contact_instances": contact_instances,
        "contact_instance_recall": sum(record["contact_instances_detected"] for record in records)
        / max(contact_instances, 1.0),
        "contact_pairs": contact_pairs,
        "contact_pair_success_rate": sum(record["contact_pair_successes"] for record in records)
        / max(contact_pairs, 1.0),
        "contact_pair_merge_rate": sum(record["contact_pair_merges"] for record in records)
        / max(contact_pairs, 1.0),
        "contact_radius": args.contact_radius,
        "boundary_tolerance": args.boundary_tolerance,
    }
    if args.mismatch_style:
        summary["style_source_policy"] = "different_domain_else_patient_else_image"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.split}_instances.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / f"{args.split}_instance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if nuceval is not None:
        (args.output_dir / f"{args.split}_nuceval_sensitivity.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in nuceval_records),
            encoding="utf-8",
        )
        nuceval_summary = {
            "split": args.split,
            "nuceval_commit": NUCEVAL_COMMIT,
            "nuceval_sha256": NUCEVAL_SHA256,
            "ground_truth_format": (
                "NuInsSeg official overlap-preserving stacked_mask MAT tensors"
            ),
            "empty_source_instance_policy": "remove zero-area slices before scoring",
            "ambiguous_overlap_threshold": 0.25,
            "match_iou": 0.5,
            "zone_widths": list(args.nuceval_zone_widths),
            "results": summarize_nuceval_records(nuceval_records),
        }
        (args.output_dir / f"{args.split}_nuceval_sensitivity_summary.json").write_text(
            json.dumps(nuceval_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
