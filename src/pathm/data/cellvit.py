from __future__ import annotations

import numpy as np


PANNUKE_NUCLEUS_TYPES = (
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
)


def majority_instance_type_counts(
    instance_map: np.ndarray,
    type_map: np.ndarray,
    num_types: int = 5,
) -> tuple[int, ...]:
    """Count instances by the majority non-background type of their pixels."""
    if instance_map.shape != type_map.shape or instance_map.ndim != 2:
        raise ValueError("Instance and type maps must be identically shaped 2D arrays")
    counts = np.zeros(num_types, dtype=np.int64)
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        values = type_map[instance_map == instance_id].astype(np.int64, copy=False)
        histogram = np.bincount(values, minlength=num_types + 1)
        histogram[0] = 0
        type_id = int(histogram.argmax())
        if not 1 <= type_id <= num_types or histogram[type_id] == 0:
            raise ValueError(f"Instance {instance_id} has no valid nucleus type")
        counts[type_id - 1] += 1
    return tuple(int(value) for value in counts)
