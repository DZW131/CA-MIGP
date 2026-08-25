from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pathm.data import (  # noqa: E402
    CRYONUSEG_ANNOTATIONS,
    CRYONUSEG_ORGANS,
    ManifestRecord,
    cryonuseg_annotation_root,
    cryonuseg_organ,
)
from pathm.data.manifest import manifest_sha256  # noqa: E402


ARCHIVE_BYTES = 172_522_035
ARCHIVE_SHA256 = "8843b3524920e0e786c66c1aa73484ed1c1ee01a5c33c20045d2b1b250c547d8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the official CryoNuSeg release")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or (
            not os.path.samefile(source, destination) and sha256(destination) != sha256(source)
        ):
            raise ValueError(f"Existing image differs from the verified source: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def write_manifest(path: Path, records: list[ManifestRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("CryoNuSeg archive size or SHA-256 does not match the pinned release")

    image_paths = sorted((source / "tissue images").glob("*.tif"))
    if len(image_paths) != 30:
        raise ValueError(f"Expected 30 CryoNuSeg images, found {len(image_paths)}")
    image_by_name = {path.name: path for path in image_paths}
    organ_counts = Counter(cryonuseg_organ(path.stem) for path in image_paths)
    if set(organ_counts) != set(CRYONUSEG_ORGANS.values()) or set(organ_counts.values()) != {3}:
        raise ValueError(f"Unexpected CryoNuSeg organ distribution: {dict(organ_counts)}")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    dataset_metadata: dict[str, object] = {
        "dataset": "CryoNuSeg",
        "archive": str(archive),
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "images": len(image_paths),
        "organs": len(organ_counts),
        "organ_counts": dict(sorted(organ_counts.items())),
        "split": "ood_test",
        "primary_reference": "ann1_round1",
        "annotations": {},
    }

    for annotation_key, annotation in CRYONUSEG_ANNOTATIONS.items():
        label_root = cryonuseg_annotation_root(source, annotation_key) / "label masks modify"
        label_paths = sorted(label_root.glob("*.tif"))
        if {path.name for path in label_paths} != set(image_by_name):
            raise ValueError(f"Image/label mismatch for CryoNuSeg {annotation_key}")
        records: list[ManifestRecord] = []
        declared_instances = 0
        visible_instances = 0
        for label_path in label_paths:
            sample_id = label_path.stem
            organ = cryonuseg_organ(sample_id)
            instance_map = np.asarray(Image.open(label_path), dtype=np.uint16)
            if instance_map.shape != (512, 512):
                raise ValueError(f"Unexpected CryoNuSeg shape for {sample_id}: {instance_map.shape}")
            declared_instances += int(instance_map.max())
            visible_instances += int(np.count_nonzero(np.unique(instance_map)))

            destination_image = output_root / "images" / organ / f"{sample_id}.tif"
            destination_label = (
                output_root / "instances" / annotation_key / organ / f"{sample_id}.npy"
            )
            hardlink_or_copy(image_by_name[label_path.name], destination_image)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            temporary_label = destination_label.with_suffix(".npy.tmp")
            with temporary_label.open("wb") as handle:
                np.save(handle, instance_map, allow_pickle=False)
            temporary_label.replace(destination_label)

            common = {
                "image": str(destination_image),
                "label": str(destination_label),
                "target": "nucleus",
                "domain": f"CryoNuSeg:{organ}",
                "patient_id": sample_id,
                "split": "ood_test",
            }
            records.extend(
                [
                    ManifestRecord(operation="detection", **common),
                    ManifestRecord(operation="segmentation", **common),
                ]
            )

        if declared_instances != annotation.declared_instances:
            raise ValueError(
                f"{annotation_key} declared-instance count is {declared_instances}, "
                f"expected {annotation.declared_instances}"
            )
        if visible_instances != annotation.visible_instances:
            raise ValueError(
                f"{annotation_key} visible-instance count is {visible_instances}, "
                f"expected {annotation.visible_instances}"
            )
        manifest_path = manifest_dir / f"cryonuseg_{annotation_key}.jsonl"
        write_manifest(manifest_path, records)
        annotation_metadata = {
            "annotator": annotation.annotator,
            "round": annotation.round,
            "reference_role": "primary" if annotation_key == "ann1_round1" else "sensitivity",
            "manifest": str(manifest_path),
            "manifest_records": len(records),
            "manifest_sha256": manifest_sha256(manifest_path),
            "declared_instances": declared_instances,
            "visible_instances": visible_instances,
        }
        metadata_path = manifest_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(annotation_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        annotations = dataset_metadata["annotations"]
        assert isinstance(annotations, dict)
        annotations[annotation_key] = annotation_metadata

    metadata_path = manifest_dir / "cryonuseg.metadata.json"
    metadata_path.write_text(
        json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(dataset_metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
