from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


STYLE_DIRECTION_MODES = {"gram", "constant", "none"}


class MultiGranularInstanceGeometryPrototypes(nn.Module):
    """Training-only instance geometry prototypes with optional direction controls."""

    def __init__(
        self,
        feature_channels: int,
        embedding_dim: int,
        num_prototypes: int = 5,
        momentum: float = 0.99,
        temperature: float = 0.1,
        max_samples_per_prototype: int = 256,
        style_direction_mode: str = "gram",
        prototype_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if num_prototypes < 2:
            raise ValueError("Geometry learning requires at least two prototypes")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Prototype momentum must be in [0, 1)")
        if temperature <= 0.0:
            raise ValueError("Prototype temperature must be positive")
        if max_samples_per_prototype <= 0:
            raise ValueError("max_samples_per_prototype must be positive")
        if style_direction_mode not in STYLE_DIRECTION_MODES:
            raise ValueError(f"Unknown style direction mode: {style_direction_mode}")

        weights = (
            torch.ones(num_prototypes, dtype=torch.float32)
            if prototype_weights is None
            else torch.as_tensor(prototype_weights, dtype=torch.float32)
        )
        if weights.shape != (num_prototypes,) or torch.any(weights <= 0):
            raise ValueError("prototype_weights must contain one positive value per prototype")

        self.num_prototypes = num_prototypes
        self.momentum = momentum
        self.temperature = temperature
        self.max_samples_per_prototype = max_samples_per_prototype
        self.style_direction_mode = style_direction_mode
        self.projection = nn.Sequential(
            nn.Linear(feature_channels, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        generator = torch.Generator().manual_seed(2718)
        initial_prototypes = torch.randn(
            num_prototypes,
            embedding_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer("prototypes", F.normalize(initial_prototypes, dim=1))
        self.register_buffer("prototype_counts", torch.zeros(num_prototypes))
        self.register_buffer("prototype_weights", weights)
        constant_direction = torch.randn(
            embedding_dim,
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer(
            "constant_style_direction",
            F.normalize(constant_direction, dim=0),
            persistent=False,
        )

    @staticmethod
    def remove_style_direction(features: Tensor, style_directions: Tensor) -> Tensor:
        if features.shape != style_directions.shape:
            raise ValueError(
                "Features and style directions must have matching sampled shapes"
            )
        unit_directions = F.normalize(style_directions.float(), dim=1).to(features.dtype)
        projections = (features * unit_directions).sum(dim=1, keepdim=True)
        return features - projections * unit_directions

    def _sample_features(
        self,
        features: Tensor,
        geometry_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sampled_features: list[Tensor] = []
        sampled_targets: list[Tensor] = []
        sampled_batch_indices: list[Tensor] = []

        for prototype_index in range(self.num_prototypes):
            positions = torch.nonzero(
                geometry_masks[:, prototype_index].bool(), as_tuple=False
            )
            if positions.shape[0] > self.max_samples_per_prototype:
                permutation = torch.randperm(positions.shape[0], device=positions.device)
                positions = positions[permutation[: self.max_samples_per_prototype]]
            if positions.numel() == 0:
                continue
            batch_indices, rows, columns = positions.unbind(dim=1)
            sampled_features.append(features[batch_indices, :, rows, columns])
            sampled_targets.append(
                torch.full(
                    (positions.shape[0],),
                    prototype_index,
                    dtype=torch.long,
                    device=features.device,
                )
            )
            sampled_batch_indices.append(batch_indices)

        if not sampled_features:
            return (
                features.new_empty((0, features.shape[1])),
                torch.empty(0, dtype=torch.long, device=features.device),
                torch.empty(0, dtype=torch.long, device=features.device),
            )
        return (
            torch.cat(sampled_features),
            torch.cat(sampled_targets),
            torch.cat(sampled_batch_indices),
        )

    @torch.no_grad()
    def _update_prototypes(self, embeddings: Tensor, targets: Tensor) -> None:
        for prototype_index in range(self.num_prototypes):
            selected = embeddings[targets == prototype_index]
            if selected.numel() == 0:
                continue
            batch_prototype = F.normalize(selected.float().mean(dim=0), dim=0)
            if self.prototype_counts[prototype_index] == 0:
                updated = batch_prototype
            else:
                updated = (
                    self.momentum * self.prototypes[prototype_index]
                    + (1.0 - self.momentum) * batch_prototype
                )
                updated = F.normalize(updated, dim=0)
            self.prototypes[prototype_index].copy_(updated)
            self.prototype_counts[prototype_index] += selected.shape[0]

    def forward(
        self,
        features: Tensor,
        geometry_masks: Tensor,
        style_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if geometry_masks.ndim != 4:
            raise ValueError("geometry_masks must have shape [B, P, H, W]")
        if geometry_masks.shape[:2] != (features.shape[0], self.num_prototypes):
            raise ValueError(
                "geometry_masks batch/prototype dimensions do not match the model"
            )
        if geometry_masks.shape[-2:] != features.shape[-2:]:
            geometry_masks = F.interpolate(
                geometry_masks.float(),
                size=features.shape[-2:],
                mode="nearest",
            ).bool()

        raw_features, targets, batch_indices = self._sample_features(
            features, geometry_masks
        )
        sample_count = features.new_tensor(targets.numel(), dtype=torch.long)
        if targets.numel() == 0:
            return features.sum() * 0.0, sample_count

        embeddings = self.projection(raw_features)
        if self.style_direction_mode == "gram":
            if style_embeddings is None:
                raise ValueError("Gram style directions require style embeddings")
            if style_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError("Style and geometry embedding dimensions must match")
            directions = style_embeddings.detach()[batch_indices]
            embeddings = self.remove_style_direction(embeddings, directions)
        elif self.style_direction_mode == "constant":
            directions = self.constant_style_direction.expand_as(embeddings)
            embeddings = self.remove_style_direction(embeddings, directions)

        normalized_embeddings = F.normalize(embeddings.float(), dim=1)
        if self.training:
            self._update_prototypes(normalized_embeddings.detach(), targets)
        normalized_prototypes = F.normalize(self.prototypes.float(), dim=1)
        logits = normalized_embeddings @ normalized_prototypes.transpose(0, 1)
        logits = logits / self.temperature
        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.prototype_weights.float(),
        )
        return loss, sample_count


# Compatibility for exploratory SO-MIGP scripts and external imports.
StyleOrthogonalGeometryPrototypes = MultiGranularInstanceGeometryPrototypes
