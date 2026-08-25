from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from evaluate_instances import load_nuinsseg_stacked_masks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit NuInsSeg overlap-preserving stacked masks used by NucEval"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-images", type=int, default=665)
    parser.add_argument("--expected-domains", type=int, default=31)
    parser.add_argument("--expected-singleton-tensors", type=int, default=5)
    parser.add_argument("--expected-empty-slices", type=int, default=1)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_stacked_masks(
    manifest_path: Path,
    raw_root: Path,
    *,
    expected_images: int,
    expected_domains: int,
    expected_singleton_tensors: int,
    expected_empty_slices: int,
) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record.get("operation") == "segmentation"]
    sample_keys = {(record["domain"], Path(record["image"]).stem) for record in records}
    if len(sample_keys) != len(records):
        raise ValueError("NuInsSeg segmentation manifest contains duplicate domain/image keys")

    domains: set[str] = set()
    total_source_slices = 0
    total_nonempty_instances = 0
    total_instance_pixels = 0
    overlap_images = 0
    overlap_pixels = 0
    singleton_tensors = 0
    empty_slices = 0
    empty_slice_sources: list[str] = []
    fingerprint_rows: list[str] = []

    for record in records:
        domain = str(record["domain"])
        image_path = str(record["image"])
        domains.add(domain)
        organ = domain.removeprefix("NuInsSeg:")
        stem = Path(image_path).stem
        binary_path = raw_root / "extracted" / organ / "mask binary" / f"{stem}.png"
        if not binary_path.is_file():
            raise ValueError(f"Missing NuInsSeg binary foreground mask: {binary_path}")
        binary = np.asarray(Image.open(binary_path).convert("L")) > 0
        masks, source, removed = load_nuinsseg_stacked_masks(
            raw_root, domain, image_path, tuple(binary.shape)
        )
        raw_stack = np.asarray(loadmat(source)["stacked_mask"])
        singleton_tensors += int(raw_stack.ndim == 2)
        source_slices = 1 if raw_stack.ndim == 2 else int(raw_stack.shape[2])
        total_source_slices += source_slices
        total_nonempty_instances += len(masks)
        empty_slices += removed
        if removed:
            empty_slice_sources.append(str(source))
        stack = np.stack(masks, axis=2)
        occupancy = stack.sum(axis=2)
        overlap = occupancy > 1
        overlap_count = int(np.count_nonzero(overlap))
        overlap_images += int(overlap_count > 0)
        overlap_pixels += overlap_count
        total_instance_pixels += int(stack.sum())
        if not np.array_equal(occupancy > 0, binary):
            mismatch = int(np.count_nonzero((occupancy > 0) != binary))
            raise ValueError(
                f"Stacked-mask union differs from binary foreground at {mismatch} pixels: {source}"
            )
        fingerprint_rows.append(f"{domain}\0{stem}\0{file_sha256(source)}")

    observed = {
        "images": len(records),
        "domains": len(domains),
        "source_slices": total_source_slices,
        "nonempty_instances": total_nonempty_instances,
        "instance_pixels": total_instance_pixels,
        "overlap_images": overlap_images,
        "overlap_pixels": overlap_pixels,
        "singleton_tensors": singleton_tensors,
        "empty_slices_removed": empty_slices,
    }
    expected = {
        "images": expected_images,
        "domains": expected_domains,
        "singleton_tensors": expected_singleton_tensors,
        "empty_slices_removed": expected_empty_slices,
    }
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise ValueError(f"NuInsSeg stacked-mask audit count mismatch: {mismatches}")
    dataset_digest = hashlib.sha256(
        ("\n".join(sorted(fingerprint_rows)) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "status": "passed",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "raw_root": str(raw_root.resolve()),
        "stacked_mask_dataset_sha256": dataset_digest,
        "expected": expected,
        "observed": observed,
        "empty_slice_sources": empty_slice_sources,
        "foreground_union_mismatch_images": 0,
        "ground_truth_format": ("NuInsSeg official overlap-preserving stacked_mask MAT tensors"),
    }


def main() -> None:
    args = parse_args()
    result = audit_stacked_masks(
        args.manifest.resolve(),
        args.raw_root.resolve(),
        expected_images=args.expected_images,
        expected_domains=args.expected_domains,
        expected_singleton_tensors=args.expected_singleton_tensors,
        expected_empty_slices=args.expected_empty_slices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
