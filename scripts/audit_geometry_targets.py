from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data import gap_boundary_masks, instance_geometry_masks  # noqa: E402
from pathm.data.manifest import load_manifest  # noqa: E402


INSTANCE_NAMES = ("rim_free", "rim_contact", "shell", "core", "near_background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit source geometry-target coverage")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mode", choices=("instance", "gap"), default="instance")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        record
        for record in load_manifest(args.manifest)
        if record.split == args.split and record.operation == "segmentation"
    ]
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        raise ValueError("No segmentation records matched the requested split")

    names = INSTANCE_NAMES if args.mode == "instance" else ("thin", "medium", "thick")
    pixel_counts = np.zeros(len(names), dtype=np.int64)
    image_counts = np.zeros(len(names), dtype=np.int64)
    total_valid_pixels = 0
    total_foreground_pixels = 0

    for record in records:
        instance_map = np.load(record.label, allow_pickle=False)
        masks = (
            instance_geometry_masks(instance_map)
            if args.mode == "instance"
            else gap_boundary_masks(instance_map)
        )
        if record.ignore is not None:
            ignore_mask = np.load(record.ignore, allow_pickle=False).astype(bool)
            masks &= ~ignore_mask[None]
            total_valid_pixels += int((~ignore_mask).sum())
            total_foreground_pixels += int(((instance_map > 0) & ~ignore_mask).sum())
        else:
            total_valid_pixels += int(instance_map.size)
            total_foreground_pixels += int((instance_map > 0).sum())
        pixel_counts += masks.sum(axis=(1, 2), dtype=np.int64)
        image_counts += masks.any(axis=(1, 2)).astype(np.int64)

    summary: dict[str, object] = {
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "mode": args.mode,
        "records": len(records),
        "total_valid_pixels": total_valid_pixels,
        "total_foreground_pixels": total_foreground_pixels,
        "prototypes": {
            name: {
                "pixels": int(pixel_counts[index]),
                "fraction_of_valid_pixels": float(pixel_counts[index] / total_valid_pixels),
                "images": int(image_counts[index]),
                "fraction_of_images": float(image_counts[index] / len(records)),
            }
            for index, name in enumerate(names)
        },
    }
    if args.mode == "instance":
        rim_total = pixel_counts[0] + pixel_counts[1]
        summary["contact_fraction_of_rim"] = float(
            pixel_counts[1] / rim_total if rim_total else 0.0
        )
        summary["foreground_partition_error"] = int(
            pixel_counts[:4].sum() - total_foreground_pixels
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
