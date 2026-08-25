from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from roifile import ImagejRoi
from skimage.draw import polygon

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data import ManifestRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the official NuInsSeg release")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def rasterize_ignore_mask(roi_paths: list[Path], shape: tuple[int, int]) -> np.ndarray:
    ignore_mask = np.zeros(shape, dtype=bool)
    for roi_path in roi_paths:
        roi = ImagejRoi.fromfile(roi_path)
        coordinates = roi.coordinates()
        if coordinates.shape[0] < 3:
            continue
        rows, columns = polygon(coordinates[:, 1], coordinates[:, 0], shape=shape)
        ignore_mask[rows, columns] = True
    return ignore_mask


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    records: list[ManifestRecord] = []
    organ_counts: Counter[str] = Counter()
    ignored_pixel_count = 0

    organs = sorted(path for path in source.iterdir() if path.is_dir())
    for organ_path in organs:
        image_root = organ_path / "tissue images"
        label_root = organ_path / "label masks modify"
        vague_root = organ_path / "vague areas" / "Imagj_zips"
        for image_path in sorted(image_root.glob("*.png")):
            sample_id = image_path.stem
            source_label = label_root / f"{sample_id}.tif"
            if not source_label.is_file():
                raise FileNotFoundError(source_label)
            instance_map = np.asarray(Image.open(source_label), dtype=np.uint16)
            if instance_map.shape != (512, 512):
                raise ValueError(f"Unexpected NuInsSeg shape for {sample_id}: {instance_map.shape}")
            roi_paths = sorted((vague_root / sample_id).glob("**/*.roi"))
            ignore_mask = rasterize_ignore_mask(roi_paths, instance_map.shape)

            organ_slug = organ_path.name.replace(" ", "_")
            destination_image = output_root / "images" / organ_slug / f"{sample_id}.png"
            destination_label = output_root / "instances" / organ_slug / f"{sample_id}.npy"
            destination_ignore = output_root / "ignore" / organ_slug / f"{sample_id}.npy"
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_ignore.parent.mkdir(parents=True, exist_ok=True)
            Image.open(image_path).convert("RGB").save(destination_image, format="PNG")
            np.save(destination_label, instance_map, allow_pickle=False)
            np.save(destination_ignore, ignore_mask, allow_pickle=False)

            common = {
                "image": str(destination_image),
                "label": str(destination_label),
                "target": "nucleus",
                "domain": f"NuInsSeg:{organ_path.name}",
                "patient_id": organ_path.name,
                "split": "ood_test",
                "ignore": str(destination_ignore),
            }
            records.extend(
                [
                    ManifestRecord(operation="detection", **common),
                    ManifestRecord(operation="segmentation", **common),
                ]
            )
            organ_counts[organ_path.name] += 1
            ignored_pixel_count += int(ignore_mask.sum())

    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")
    temporary_manifest.replace(args.manifest)
    metadata = {
        "dataset": "NuInsSeg",
        "split": "ood_test",
        "images": len(records) // 2,
        "manifest_records": len(records),
        "wsi_domain_groups": len(organ_counts),
        "organ_counts": dict(organ_counts),
        "ignored_pixels": ignored_pixel_count,
        "source": str(source),
    }
    metadata_path = args.manifest.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
