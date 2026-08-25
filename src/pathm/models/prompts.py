from __future__ import annotations

import torch
from torch import Tensor, nn


STYLE_PROMPT_MODES = {"gram", "diagonal", "constant"}


class GramStyleEncoder(nn.Module):
    """Encode full, diagonal, or image-independent Gram statistics."""

    def __init__(
        self,
        in_channels: int,
        style_channels: int,
        prompt_dim: int,
        mode: str = "gram",
    ) -> None:
        super().__init__()
        if mode not in STYLE_PROMPT_MODES:
            raise ValueError(f"Unknown style prompt mode: {mode}")
        self.mode = mode
        self.project = nn.Conv2d(in_channels, style_channels, kernel_size=1, bias=False)
        self.normalization = nn.LayerNorm(style_channels * style_channels)
        self.encoder = nn.Sequential(
            nn.Linear(style_channels * style_channels, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )
        generator = torch.Generator().manual_seed(1729)
        constant_features = torch.randn(
            1, in_channels, 8, 8, generator=generator, dtype=torch.float32
        )
        constant_features = (constant_features - constant_features.mean()) / (
            constant_features.std() + 1e-6
        )
        self.register_buffer("constant_features", constant_features, persistent=False)

    def forward(self, features: Tensor) -> Tensor:
        style_features = features
        if self.mode == "constant":
            style_features = self.constant_features.expand(features.shape[0], -1, -1, -1)
        projected = self.project(style_features)
        batch, channels, height, width = projected.shape
        flattened = projected.reshape(batch, channels, height * width)
        gram = torch.bmm(flattened, flattened.transpose(1, 2))
        gram = gram / float(channels * height * width)
        if self.mode == "diagonal":
            gram = torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))
        return self.encoder(self.normalization(gram.flatten(1)))


class FactorizedTaskPrompt(nn.Module):
    """Compose operation, target, and image-style information."""

    def __init__(
        self,
        num_operations: int,
        num_targets: int,
        feature_channels: int,
        style_channels: int,
        prompt_dim: int,
        use_style: bool = True,
        factorized: bool = True,
        style_mode: str = "gram",
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.factorized = factorized
        self.operation_embedding = (
            nn.Embedding(num_operations, prompt_dim) if factorized else None
        )
        self.target_embedding = nn.Embedding(num_targets, prompt_dim) if factorized else None
        self.joint_embedding = (
            None if factorized else nn.Embedding(num_operations * num_targets, prompt_dim)
        )
        self.style_encoder = (
            GramStyleEncoder(
                feature_channels,
                style_channels,
                prompt_dim,
                mode=style_mode,
            )
            if use_style
            else None
        )
        component_count = (2 if factorized else 1) + int(use_style)
        self.fusion = nn.Sequential(
            nn.LayerNorm(prompt_dim * component_count),
            nn.Linear(prompt_dim * component_count, prompt_dim * 2),
            nn.GELU(),
            nn.Linear(prompt_dim * 2, prompt_dim),
        )

    def forward(
        self,
        features: Tensor,
        operation_ids: Tensor,
        target_ids: Tensor,
        style_features: Tensor | None = None,
    ) -> Tensor:
        prompt, _ = self.compose(
            features,
            operation_ids,
            target_ids,
            style_features=style_features,
        )
        return prompt

    def compose(
        self,
        features: Tensor,
        operation_ids: Tensor,
        target_ids: Tensor,
        style_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        components: list[Tensor]
        if self.factorized:
            assert self.operation_embedding is not None and self.target_embedding is not None
            components = [
                self.operation_embedding(operation_ids),
                self.target_embedding(target_ids),
            ]
        else:
            assert self.joint_embedding is not None
            joint_ids = operation_ids * self.num_targets + target_ids
            components = [self.joint_embedding(joint_ids)]
        style_embedding = None
        if self.style_encoder is not None:
            style_embedding = self.style_encoder(
                features if style_features is None else style_features
            )
            components.append(style_embedding)
        return self.fusion(torch.cat(components, dim=-1)), style_embedding


class ChannelPrompt(nn.Module):
    def __init__(self, channels: int, prompt_dim: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.gate = nn.Sequential(
            nn.Linear(channels + prompt_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor, prompt: Tensor) -> tuple[Tensor, Tensor]:
        pooled = features.mean(dim=(-2, -1))
        gate = self.gate(torch.cat((pooled, prompt), dim=-1))[:, :, None, None]
        return features * (1.0 + gate), gate


class SpatialPrompt(nn.Module):
    def __init__(self, channels: int, prompt_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(prompt_dim, channels)
        self.refine = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.scale = channels**-0.5

    def forward(self, features: Tensor, prompt: Tensor) -> tuple[Tensor, Tensor]:
        query = self.query(prompt)[:, :, None, None]
        response = (features * query).sum(dim=1, keepdim=True) * self.scale
        gate = torch.sigmoid(self.refine(response))
        return features * (1.0 + gate), gate


class HierarchicalPromptModulator(nn.Module):
    def __init__(
        self,
        channels: int,
        prompt_dim: int,
        use_channel: bool = True,
        use_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.channel_prompt = ChannelPrompt(channels, prompt_dim) if use_channel else None
        self.spatial_prompt = SpatialPrompt(channels, prompt_dim) if use_spatial else None
        branch_count = int(use_channel) + int(use_spatial)
        self.fuse = (
            nn.Sequential(
                nn.Conv2d(channels * (1 + branch_count), channels, kernel_size=1, bias=False),
                nn.GroupNorm(_group_count(channels), channels),
                nn.GELU(),
            )
            if branch_count
            else None
        )

    def forward(self, features: Tensor, prompt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        branches = [features]
        if self.channel_prompt is not None:
            channel_features, channel_gate = self.channel_prompt(features, prompt)
            branches.append(channel_features)
        else:
            channel_gate = features.new_zeros(features.shape[0], features.shape[1], 1, 1)
        if self.spatial_prompt is not None:
            spatial_features, spatial_gate = self.spatial_prompt(features, prompt)
            branches.append(spatial_features)
        else:
            spatial_gate = features.new_zeros(
                features.shape[0], 1, features.shape[2], features.shape[3]
            )
        fused = features
        if self.fuse is not None:
            fused = features + self.fuse(torch.cat(branches, dim=1))
        return fused, channel_gate, spatial_gate


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
