from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data import ManifestRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PanNuke parquet shards")
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_image(value: dict[str, bytes | str | None]) -> Image.Image:
    raw = value.get("bytes")
    if not isinstance(raw, bytes):
        raise ValueError("Expected embedded image bytes in PanNuke parquet")
    return Image.open(io.BytesIO(raw))


def main() -> None:
    args = parse_args()
    inputs = sorted(path.resolve() for path in args.input)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    images_root = (args.output_root / "images").resolve()
    labels_root = (args.output_root / "instances").resolve()
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    fold_splits = {1: "train", 2: "val", 3: "test"}
    records: list[ManifestRecord] = []
    tissue_counts: Counter[str] = Counter()
    image_count = 0

    for parquet_path in inputs:
        parquet_file = pq.ParquetFile(parquet_path)
        columns = ["image", "inst_map", "tissue_name", "fold", "sample_id"]
        for batch in parquet_file.iter_batches(batch_size=args.batch_size, columns=columns):
            rows = batch.to_pylist()
            for row in rows:
                fold = int(row["fold"])
                if fold not in fold_splits:
                    raise ValueError(f"Unexpected PanNuke fold: {fold}")
                sample_id = str(row["sample_id"])
                fold_name = f"fold{fold}"
                image_path = images_root / fold_name / f"{sample_id}.png"
                label_path = labels_root / fold_name / f"{sample_id}.npy"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)

                image = decode_image(row["image"]).convert("RGB")
                instance_map = np.asarray(decode_image(row["inst_map"]), dtype=np.uint16)
                image.save(image_path, format="PNG", optimize=False)
                np.save(label_path, instance_map, allow_pickle=False)

                common = {
                    "image": str(image_path),
                    "label": str(label_path),
                    "target": "nucleus",
                    "domain": "PanNuke",
                    # PanNuke mirror has no patient/WSI ID; this is a patch-level group surrogate.
                    "patient_id": sample_id,
                    "split": fold_splits[fold],
                }
                records.extend(
                    [
                        ManifestRecord(operation="detection", **common),
                        ManifestRecord(operation="segmentation", **common),
                    ]
                )
                tissue_counts[str(row["tissue_name"])] += 1
                image_count += 1
                if args.max_images is not None and image_count >= args.max_images:
                    break
            if args.max_images is not None and image_count >= args.max_images:
                break
        if args.max_images is not None and image_count >= args.max_images:
            break

    temporary_manifest = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")
    temporary_manifest.replace(args.manifest)

    metadata = {
        "dataset": "PanNuke",
        "source_mirror": "MedOtter/PanNuke",
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "fold_mapping": {str(key): value for key, value in fold_splits.items()},
        "images": image_count,
        "manifest_records": len(records),
        "tissue_counts": dict(sorted(tissue_counts.items())),
        "patient_wsi_ids_available": False,
        "grouping_surrogate": "sample_id",
    }
    metadata_path = args.manifest.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"prepared_images={image_count}")
    print(f"manifest_records={len(records)}")
    print(f"manifest={args.manifest.resolve()}")
    print(f"metadata={metadata_path.resolve()}")


if __name__ == "__main__":
    main()
