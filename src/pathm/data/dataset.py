from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.color import hed2rgb, rgb2hed
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import ManifestRecord, load_manifest


OPERATION_IDS = {"detection": 0, "segmentation": 1}
TARGET_IDS = {"nucleus": 0, "gland": 1, "tissue": 2, "tumor": 3, "other": 4}
GEOMETRY_TARGET_MODES = {"instance", "instance_no_contact", "gap"}


def style_donor_mapping(
    records: Sequence[ManifestRecord],
) -> dict[str, ManifestRecord]:
    """Assign every unique image a deterministic donor from another available group."""
    unique_records: dict[str, ManifestRecord] = {}
    for record in records:
        unique_records.setdefault(record.image, record)
    images = list(unique_records)
    if len(images) < 2:
        raise ValueError("Style mismatch evaluation requires at least two unique images")

    entries = list(unique_records.values())
    if len({record.domain for record in entries}) > 1:
        group_field = "domain"
    elif len({record.patient_id for record in entries}) > 1:
        group_field = "patient_id"
    else:
        group_field = "image"

    def group(record: ManifestRecord) -> str:
        return str(getattr(record, group_field))

    grouped: dict[str, list[ManifestRecord]] = {}
    for record in entries:
        grouped.setdefault(group(record), []).append(record)
    group_names = list(grouped)
    if len(group_names) < 2:
        raise ValueError("Could not identify distinct style-donor groups")

    donors: dict[str, ManifestRecord] = {}
    for group_index, group_name in enumerate(group_names):
        donor_group = grouped[group_names[(group_index + 1) % len(group_names)]]
        for image_index, record in enumerate(grouped[group_name]):
            donor = donor_group[image_index % len(donor_group)]
            if donor.image == record.image or group(donor) == group(record):
                raise ValueError(f"Invalid style donor for {record.image}")
            donors[record.image] = donor
    return donors


def apply_hed_stain_scale(
    image: np.ndarray, hematoxylin_scale: float = 1.0, eosin_scale: float = 1.0
) -> np.ndarray:
    if hematoxylin_scale <= 0 or eosin_scale <= 0:
        raise ValueError("HED stain scales must be positive")
    if hematoxylin_scale == 1.0 and eosin_scale == 1.0:
        return image.astype(np.float32, copy=True)
    hed = rgb2hed(np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0))
    hed[..., 0] *= hematoxylin_scale
    hed[..., 1] *= eosin_scale
    return np.clip(hed2rgb(hed), 0.0, 1.0).astype(np.float32) * 255.0


def apply_random_hed_stain_jitter(image: np.ndarray, strength: float) -> np.ndarray:
    if not 0.0 <= strength < 1.0:
        raise ValueError("HED stain jitter strength must be in [0, 1)")
    if strength == 0.0:
        return image.astype(np.float32, copy=True)
    hematoxylin_scale = 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * strength
    eosin_scale = 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * strength
    return apply_hed_stain_scale(image, hematoxylin_scale, eosin_scale)


def instance_centroids(instance_map: np.ndarray) -> list[tuple[int, int]]:
    if instance_map.ndim != 2:
        raise ValueError(f"Expected a 2D instance map, got shape {instance_map.shape}")
    centroids: list[tuple[int, int]] = []
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        rows, columns = np.nonzero(instance_map == instance_id)
        if rows.size:
            centroids.append((int(np.rint(rows.mean())), int(np.rint(columns.mean()))))
    return centroids


def detection_heatmap(instance_map: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    height, width = instance_map.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    for row, column in instance_centroids(instance_map):
        top, bottom = max(row - radius, 0), min(row + radius + 1, height)
        left, right = max(column - radius, 0), min(column + radius + 1, width)
        yy = np.arange(top, bottom, dtype=np.float32)[:, None]
        xx = np.arange(left, right, dtype=np.float32)[None, :]
        gaussian = np.exp(-((yy - row) ** 2 + (xx - column) ** 2) / (2.0 * sigma**2))
        heatmap[top:bottom, left:right] = np.maximum(
            heatmap[top:bottom, left:right], gaussian
        )
    return heatmap


def instance_geometry_masks(
    instance_map: np.ndarray,
    rim_threshold: float = 0.15,
    shell_threshold: float = 0.35,
    contact_radius: float = 3.0,
    near_background_radius: float = 3.0,
    split_contacts: bool = True,
) -> np.ndarray:
    """Build scale-normalized radial masks, optionally splitting contact rims."""
    if instance_map.ndim != 2:
        raise ValueError(f"Expected a 2D instance map, got shape {instance_map.shape}")
    if not 0.0 < rim_threshold < shell_threshold:
        raise ValueError("Geometry thresholds must satisfy 0 < rim < shell")
    if contact_radius <= 0.0 or near_background_radius <= 0.0:
        raise ValueError("Geometry radii must be positive")

    prototype_count = 5 if split_contacts else 4
    masks = np.zeros((prototype_count, *instance_map.shape), dtype=bool)
    foreground = instance_map > 0
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        instance = instance_map == instance_id
        area = int(instance.sum())
        if area == 0:
            continue
        equivalent_radius = np.sqrt(area / np.pi)
        inward_distance = np.maximum(ndimage.distance_transform_edt(instance) - 1.0, 0.0)
        normalized_depth = inward_distance / max(equivalent_radius, 1e-6)
        rim = instance & (normalized_depth < rim_threshold)
        shell = instance & (normalized_depth >= rim_threshold) & (
            normalized_depth < shell_threshold
        )
        core = instance & (normalized_depth >= shell_threshold)

        if split_contacts:
            other_instances = foreground & ~instance
            if other_instances.any():
                distance_to_other = ndimage.distance_transform_edt(~other_instances)
                contact = rim & (distance_to_other <= contact_radius)
            else:
                contact = np.zeros_like(instance)
            masks[0] |= rim & ~contact
            masks[1] |= contact
            masks[2] |= shell
            masks[3] |= core
        else:
            masks[0] |= rim
            masks[1] |= shell
            masks[2] |= core

    if foreground.any():
        distance_to_foreground = ndimage.distance_transform_edt(~foreground)
        masks[-1] = (~foreground) & (distance_to_foreground <= near_background_radius)
    return masks


def gap_boundary_masks(
    instance_map: np.ndarray,
    kernel_sizes: Sequence[int] = (1, 2, 3),
) -> np.ndarray:
    """Natural-image GAP control using nested fixed-pixel boundary masks."""
    if instance_map.ndim != 2:
        raise ValueError(f"Expected a 2D instance map, got shape {instance_map.shape}")
    if len(kernel_sizes) < 2 or any(int(size) <= 0 for size in kernel_sizes):
        raise ValueError("GAP kernel sizes must contain at least two positive integers")
    foreground = instance_map > 0
    masks = []
    for size in kernel_sizes:
        dilated = ndimage.binary_dilation(foreground, iterations=int(size))
        eroded = ndimage.binary_erosion(foreground, iterations=int(size))
        masks.append(dilated & ~eroded)
    return np.stack(masks, axis=0)


class ManifestTaskDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        manifest_path: str | Path,
        splits: str | Sequence[str],
        augment: bool = False,
        heatmap_sigma: float = 2.0,
        color_jitter_strength: float = 0.0,
        hed_stain_jitter_strength: float = 0.0,
        max_records: int | None = None,
        operations: Sequence[str] | None = None,
        stain_h_scale: float = 1.0,
        stain_e_scale: float = 1.0,
        mismatched_style: bool = False,
        include_geometry_targets: bool = False,
        geometry_target_mode: str = "instance",
        geometry_rim_threshold: float = 0.15,
        geometry_shell_threshold: float = 0.35,
        geometry_contact_radius: float = 3.0,
        geometry_near_background_radius: float = 3.0,
        geometry_gap_kernel_sizes: Sequence[int] = (1, 2, 3),
    ) -> None:
        split_set = {splits} if isinstance(splits, str) else set(splits)
        operation_set = set(operations) if operations is not None else None
        self.records = [
            record
            for record in load_manifest(manifest_path)
            if record.split in split_set
            and (operation_set is None or record.operation in operation_set)
        ]
        if max_records is not None:
            self.records = self.records[:max_records]
        if not self.records:
            raise ValueError(f"No records found for splits {sorted(split_set)}")
        self.augment = augment
        self.heatmap_sigma = heatmap_sigma
        self.color_jitter_strength = color_jitter_strength
        if not 0.0 <= hed_stain_jitter_strength < 1.0:
            raise ValueError("HED stain jitter strength must be in [0, 1)")
        self.hed_stain_jitter_strength = hed_stain_jitter_strength
        if stain_h_scale <= 0 or stain_e_scale <= 0:
            raise ValueError("HED stain scales must be positive")
        self.stain_h_scale = stain_h_scale
        self.stain_e_scale = stain_e_scale
        if augment and mismatched_style:
            raise ValueError("Mismatched style donors are only supported without augmentation")
        self.style_donors = style_donor_mapping(self.records) if mismatched_style else None
        if geometry_target_mode not in GEOMETRY_TARGET_MODES:
            raise ValueError(f"Unknown geometry target mode: {geometry_target_mode}")
        self.include_geometry_targets = include_geometry_targets
        self.geometry_target_mode = geometry_target_mode
        self.geometry_rim_threshold = geometry_rim_threshold
        self.geometry_shell_threshold = geometry_shell_threshold
        self.geometry_contact_radius = geometry_contact_radius
        self.geometry_near_background_radius = geometry_near_background_radius
        self.geometry_gap_kernel_sizes = tuple(int(size) for size in geometry_gap_kernel_sizes)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        image = np.asarray(Image.open(record.image).convert("RGB"), dtype=np.float32).copy()
        image = apply_hed_stain_scale(image, self.stain_h_scale, self.stain_e_scale)
        instance_map = np.load(record.label, allow_pickle=False)
        ignore_mask = (
            np.load(record.ignore, allow_pickle=False).astype(bool)
            if record.ignore is not None
            else np.zeros(instance_map.shape, dtype=bool)
        )
        if image.shape[:2] != instance_map.shape:
            raise ValueError(
                f"Image/label shape mismatch for {record.image}: {image.shape[:2]} "
                f"versus {instance_map.shape}"
            )
        if record.operation == "detection":
            target = detection_heatmap(instance_map, self.heatmap_sigma)
        else:
            target = (instance_map > 0).astype(np.float32)
        geometry_masks = None
        if self.include_geometry_targets:
            if self.geometry_target_mode == "instance":
                prototype_count = 5
            elif self.geometry_target_mode == "instance_no_contact":
                prototype_count = 4
            else:
                prototype_count = len(self.geometry_gap_kernel_sizes)
            if record.operation == "detection":
                geometry_masks = np.zeros(
                    (prototype_count, *instance_map.shape), dtype=bool
                )
            elif self.geometry_target_mode in {"instance", "instance_no_contact"}:
                geometry_masks = instance_geometry_masks(
                    instance_map,
                    rim_threshold=self.geometry_rim_threshold,
                    shell_threshold=self.geometry_shell_threshold,
                    contact_radius=self.geometry_contact_radius,
                    near_background_radius=self.geometry_near_background_radius,
                    split_contacts=self.geometry_target_mode == "instance",
                )
            else:
                geometry_masks = gap_boundary_masks(
                    instance_map,
                    kernel_sizes=self.geometry_gap_kernel_sizes,
                )
            geometry_masks &= ~ignore_mask[None]

        if self.augment:
            image = apply_random_hed_stain_jitter(image, self.hed_stain_jitter_strength)
            if torch.rand(()) < 0.5:
                image = np.flip(image, axis=1).copy()
                target = np.flip(target, axis=1).copy()
                ignore_mask = np.flip(ignore_mask, axis=1).copy()
                if geometry_masks is not None:
                    geometry_masks = np.flip(geometry_masks, axis=2).copy()
            if torch.rand(()) < 0.5:
                image = np.flip(image, axis=0).copy()
                target = np.flip(target, axis=0).copy()
                ignore_mask = np.flip(ignore_mask, axis=0).copy()
                if geometry_masks is not None:
                    geometry_masks = np.flip(geometry_masks, axis=1).copy()
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                image = np.rot90(image, rotations, axes=(0, 1)).copy()
                target = np.rot90(target, rotations, axes=(0, 1)).copy()
                ignore_mask = np.rot90(ignore_mask, rotations, axes=(0, 1)).copy()
                if geometry_masks is not None:
                    geometry_masks = np.rot90(
                        geometry_masks, rotations, axes=(1, 2)
                    ).copy()
            if self.color_jitter_strength > 0:
                strength = self.color_jitter_strength
                brightness = 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * strength
                contrast = 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * strength
                saturation = 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * strength
                image = image * brightness
                image_mean = image.mean(axis=(0, 1), keepdims=True)
                image = (image - image_mean) * contrast + image_mean
                grayscale = (
                    image[..., 0:1] * 0.299
                    + image[..., 1:2] * 0.587
                    + image[..., 2:3] * 0.114
                )
                image = grayscale + (image - grayscale) * saturation
                image = np.clip(image, 0.0, 255.0)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        sample: dict[str, Tensor | str] = {
            "image": image_tensor,
            "target": torch.from_numpy(target[None]),
            "valid_mask": torch.from_numpy((~ignore_mask)[None]),
            "operation_id": torch.tensor(OPERATION_IDS[record.operation], dtype=torch.long),
            "target_id": torch.tensor(TARGET_IDS[record.target], dtype=torch.long),
            "operation": record.operation,
            "sample_id": record.patient_id,
            "domain": record.domain,
            "image_path": record.image,
            "label_path": record.label,
            "ignore_path": record.ignore or "",
        }
        if geometry_masks is not None:
            sample["geometry_masks"] = torch.from_numpy(geometry_masks)
        if self.style_donors is not None:
            donor = self.style_donors[record.image]
            donor_image = np.asarray(
                Image.open(donor.image).convert("RGB"), dtype=np.float32
            ).copy()
            donor_image = apply_hed_stain_scale(
                donor_image, self.stain_h_scale, self.stain_e_scale
            )
            if donor_image.shape != image.shape:
                raise ValueError(
                    f"Style donor shape mismatch for {record.image}: "
                    f"{donor_image.shape} versus {image.shape}"
                )
            donor_tensor = torch.from_numpy(donor_image.transpose(2, 0, 1)) / 255.0
            sample.update(
                {
                    "style_image": (donor_tensor - 0.5) / 0.5,
                    "style_source_sample_id": donor.patient_id,
                    "style_source_domain": donor.domain,
                    "style_source_image": donor.image,
                }
            )
        return sample
