from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from pathm.data import (
    ManifestTaskDataset,
    apply_hed_stain_scale,
    apply_random_hed_stain_jitter,
    detection_heatmap,
    gap_boundary_masks,
    instance_geometry_masks,
    instance_centroids,
    style_donor_mapping,
)
from pathm.data.manifest import ManifestRecord


def test_instance_centroids_and_heatmap() -> None:
    instance_map = np.zeros((16, 16), dtype=np.uint16)
    instance_map[2:5, 3:6] = 1
    instance_map[10:12, 11:15] = 2
    assert instance_centroids(instance_map) == [(3, 4), (10, 12)]
    heatmap = detection_heatmap(instance_map, sigma=1.0)
    assert heatmap.shape == instance_map.shape
    assert heatmap[3, 4] == 1.0
    assert heatmap[10, 12] == 1.0


def test_instance_geometry_masks_cover_nuclei_and_find_contacts() -> None:
    instance_map = np.zeros((18, 18), dtype=np.uint16)
    instance_map[4:14, 2:8] = 1
    instance_map[4:14, 8:14] = 2

    masks = instance_geometry_masks(instance_map, contact_radius=1.5)

    assert masks.shape == (5, 18, 18)
    np.testing.assert_array_equal(masks[:4].sum(axis=0) > 0, instance_map > 0)
    assert np.all(masks[:4].sum(axis=0) <= 1)
    assert masks[1, :, 7:9].any()
    assert masks[4].any()
    assert not np.any(masks[4] & (instance_map > 0))


def test_instance_geometry_masks_can_merge_contact_and_free_rims() -> None:
    instance_map = np.zeros((18, 18), dtype=np.uint16)
    instance_map[4:14, 2:8] = 1
    instance_map[4:14, 8:14] = 2

    split = instance_geometry_masks(instance_map, contact_radius=1.5)
    merged = instance_geometry_masks(
        instance_map,
        contact_radius=1.5,
        split_contacts=False,
    )

    assert merged.shape == (4, 18, 18)
    np.testing.assert_array_equal(merged[0], split[0] | split[1])
    np.testing.assert_array_equal(merged[1], split[2])
    np.testing.assert_array_equal(merged[2], split[3])
    np.testing.assert_array_equal(merged[3], split[4])


def test_gap_boundary_masks_are_nested() -> None:
    instance_map = np.zeros((21, 21), dtype=np.uint16)
    instance_map[6:15, 6:15] = 1

    masks = gap_boundary_masks(instance_map, kernel_sizes=(1, 2, 3))

    assert masks.shape == (3, 21, 21)
    assert masks[0].sum() < masks[1].sum() < masks[2].sum()
    assert np.all(masks[0] <= masks[1])
    assert np.all(masks[1] <= masks[2])


def test_hed_stain_scale_is_identity_at_one_and_changes_color() -> None:
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[..., 0] = 170
    image[..., 1] = 90
    image[..., 2] = 145

    identity = apply_hed_stain_scale(image)
    shifted = apply_hed_stain_scale(image, hematoxylin_scale=1.3, eosin_scale=0.7)

    np.testing.assert_array_equal(identity, image)
    assert shifted.dtype == np.float32
    assert shifted.min() >= 0.0
    assert shifted.max() <= 255.0
    assert not np.allclose(shifted, image)


def test_random_hed_stain_jitter_is_seeded_and_bounded() -> None:
    image = np.full((8, 8, 3), (170, 90, 145), dtype=np.float32)

    torch.manual_seed(7)
    first = apply_random_hed_stain_jitter(image, strength=0.3)
    torch.manual_seed(7)
    second = apply_random_hed_stain_jitter(image, strength=0.3)

    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0.0
    assert first.max() <= 255.0
    assert not np.allclose(first, image)


def test_manifest_dataset_emits_both_tasks(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    label_path = tmp_path / "instances.npy"
    ignore_path = tmp_path / "ignore.npy"
    manifest_path = tmp_path / "manifest.jsonl"
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(image_path)
    instance_map = np.zeros((16, 16), dtype=np.uint16)
    instance_map[4:8, 5:9] = 1
    np.save(label_path, instance_map, allow_pickle=False)
    ignore_mask = np.zeros((16, 16), dtype=bool)
    ignore_mask[0, 0] = True
    np.save(ignore_path, ignore_mask, allow_pickle=False)
    common = {
        "image": str(image_path),
        "label": str(label_path),
        "target": "nucleus",
        "domain": "PanNuke",
        "patient_id": "sample-1",
        "split": "train",
        "ignore": str(ignore_path),
    }
    rows = [
        {**common, "operation": "detection"},
        {**common, "operation": "segmentation"},
    ]
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    dataset = ManifestTaskDataset(
        manifest_path,
        "train",
        include_geometry_targets=True,
    )
    detection = dataset[0]
    segmentation = dataset[1]
    assert detection["image"].shape == (3, 16, 16)
    assert detection["target"].max().item() == 1.0
    assert detection["operation_id"].item() == 0
    assert detection["geometry_masks"].shape == (5, 16, 16)
    assert not detection["geometry_masks"].any()
    assert segmentation["target"].sum().item() == 16
    assert segmentation["operation_id"].item() == 1
    assert segmentation["geometry_masks"].any()
    assert not segmentation["geometry_masks"][:, 0, 0].any()
    assert not segmentation["valid_mask"][0, 0, 0]


def test_style_donors_are_deterministic_and_cross_domain() -> None:
    records = [
        ManifestRecord(
            image=f"image-{index}.png",
            label=f"label-{index}.npy",
            operation="segmentation",
            target="nucleus",
            domain=domain,
            patient_id=f"patient-{index}",
            split="ood_test",
        )
        for index, domain in enumerate(("a", "a", "b", "c"))
    ]

    first = style_donor_mapping(records)
    second = style_donor_mapping(records)

    assert {key: value.image for key, value in first.items()} == {
        key: value.image for key, value in second.items()
    }
    assert all(
        donor.image != image
        and donor.domain != next(record.domain for record in records if record.image == image)
        for image, donor in first.items()
    )


def test_manifest_dataset_loads_cross_domain_style_image(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    for index, domain in enumerate(("a", "b", "c")):
        image_path = tmp_path / f"image-{index}.png"
        label_path = tmp_path / f"label-{index}.npy"
        Image.fromarray(np.full((8, 8, 3), index * 80, dtype=np.uint8)).save(image_path)
        np.save(label_path, np.zeros((8, 8), dtype=np.uint16), allow_pickle=False)
        rows.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "operation": "segmentation",
                "target": "nucleus",
                "domain": domain,
                "patient_id": f"patient-{index}",
                "split": "ood_test",
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    dataset = ManifestTaskDataset(
        manifest_path,
        "ood_test",
        operations=["segmentation"],
        mismatched_style=True,
    )
    sample = dataset[0]

    assert sample["style_source_domain"] != sample["domain"]
    assert sample["style_source_image"] != sample["image_path"]
    assert sample["style_image"].shape == sample["image"].shape
