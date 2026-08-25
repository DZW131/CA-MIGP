from __future__ import annotations

from torch import nn

from .baselines import MultiHeadUNet, SpecialistUNet
from .geometry import (
    MultiGranularInstanceGeometryPrototypes,
    StyleOrthogonalGeometryPrototypes,
)
from .network import PathMUnified


def build_model(config: dict[str, object]) -> nn.Module:
    parameters = dict(config)
    architecture = parameters.pop("architecture", "prompt_unified")
    if architecture == "prompt_unified":
        return PathMUnified(**parameters)  # type: ignore[arg-type]
    if architecture == "multihead_unet":
        return MultiHeadUNet(**parameters)  # type: ignore[arg-type]
    if architecture == "specialist_unet":
        return SpecialistUNet(**parameters)  # type: ignore[arg-type]
    raise ValueError(f"Unknown model architecture: {architecture}")


__all__ = [
    "MultiHeadUNet",
    "PathMUnified",
    "SpecialistUNet",
    "MultiGranularInstanceGeometryPrototypes",
    "StyleOrthogonalGeometryPrototypes",
    "build_model",
]
