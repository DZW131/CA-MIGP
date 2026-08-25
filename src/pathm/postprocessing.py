from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import watershed

from pathm.metrics import heatmap_points


def seeded_watershed_instances(
    segmentation_probability: np.ndarray,
    detection_heatmap: np.ndarray,
    segmentation_threshold: float = 0.5,
    detection_threshold: float = 0.1,
    detection_minimum_distance: int = 3,
) -> np.ndarray:
    if segmentation_probability.shape != detection_heatmap.shape:
        raise ValueError("Segmentation and detection maps must have identical shapes")
    foreground = segmentation_probability >= segmentation_threshold
    markers = np.zeros(foreground.shape, dtype=np.int32)
    points = heatmap_points(
        torch.from_numpy(detection_heatmap),
        threshold=detection_threshold,
        minimum_distance=detection_minimum_distance,
    )
    marker_id = 1
    for row_value, column_value in points:
        row, column = int(row_value), int(column_value)
        if foreground[row, column] and markers[row, column] == 0:
            markers[row, column] = marker_id
            marker_id += 1

    connected, connected_count = ndimage.label(foreground)
    for component_id in range(1, connected_count + 1):
        component = connected == component_id
        if np.any(markers[component]):
            continue
        distance = ndimage.distance_transform_edt(component)
        row, column = np.unravel_index(np.argmax(distance), distance.shape)
        markers[row, column] = marker_id
        marker_id += 1

    if marker_id == 1:
        return markers.astype(np.uint16)
    distance = ndimage.distance_transform_edt(foreground)
    instances = watershed(-distance, markers=markers, mask=foreground)
    return instances.astype(np.uint16)
