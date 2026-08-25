from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .prompts import FactorizedTaskPrompt, HierarchicalPromptModulator, _group_count
from .geometry import MultiGranularInstanceGeometryPrototypes


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DynamicOutputHead(nn.Module):
    """Generate one 1x1 output kernel per sample from the fused prompt."""

    def __init__(self, feature_channels: int, prompt_dim: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.parameter_generator = nn.Linear(prompt_dim, feature_channels + 1)

    def forward(self, features: Tensor, prompt: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        if channels != self.feature_channels:
            raise ValueError(f"Expected {self.feature_channels} channels, got {channels}")
        parameters = self.parameter_generator(prompt)
        weight = parameters[:, :channels].reshape(batch, channels, 1, 1)
        bias = parameters[:, channels]
        grouped_features = features.reshape(1, batch * channels, height, width)
        logits = F.conv2d(grouped_features, weight, bias=bias, groups=batch)
        return logits.reshape(batch, 1, height, width)


class PathMUnified(nn.Module):
    """Minimal unified model used to validate the prompt mechanism."""

    def __init__(
        self,
        base_channels: int = 32,
        prompt_dim: int = 128,
        style_channels: int = 32,
        num_operations: int = 2,
        num_targets: int = 5,
        use_style_prompt: bool = True,
        style_prompt_mode: str = "gram",
        factorized_task_prompt: bool = True,
        use_channel_prompt: bool = True,
        use_spatial_prompt: bool = True,
        use_geometry_prototypes: bool = False,
        geometry_num_prototypes: int = 5,
        geometry_momentum: float = 0.99,
        geometry_temperature: float = 0.1,
        geometry_max_samples_per_prototype: int = 256,
        geometry_style_direction_mode: str = "gram",
        geometry_prototype_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoder_blocks = nn.ModuleList(
            [
                ConvBlock(3, channels[0]),
                ConvBlock(channels[0], channels[1]),
                ConvBlock(channels[1], channels[2]),
                ConvBlock(channels[2], channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.task_prompt = FactorizedTaskPrompt(
            num_operations,
            num_targets,
            channels[-1],
            style_channels,
            prompt_dim,
            use_style=use_style_prompt,
            factorized=factorized_task_prompt,
            style_mode=style_prompt_mode,
        )
        self.modulator = HierarchicalPromptModulator(
            channels[-1],
            prompt_dim,
            use_channel=use_channel_prompt,
            use_spatial=use_spatial_prompt,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                ConvBlock(channels[3] + channels[2], channels[2]),
                ConvBlock(channels[2] + channels[1], channels[1]),
                ConvBlock(channels[1] + channels[0], channels[0]),
            ]
        )
        self.output_head = DynamicOutputHead(channels[0], prompt_dim)
        if (
            use_geometry_prototypes
            and geometry_style_direction_mode == "gram"
            and not use_style_prompt
        ):
            raise ValueError("Gram-orthogonal geometry prototypes require a style prompt")
        self.geometry_prototypes = (
            MultiGranularInstanceGeometryPrototypes(
                feature_channels=channels[0],
                embedding_dim=prompt_dim,
                num_prototypes=geometry_num_prototypes,
                momentum=geometry_momentum,
                temperature=geometry_temperature,
                max_samples_per_prototype=geometry_max_samples_per_prototype,
                style_direction_mode=geometry_style_direction_mode,
                prototype_weights=geometry_prototype_weights,
            )
            if use_geometry_prototypes
            else None
        )

    def forward(
        self,
        images: Tensor,
        operation_ids: Tensor,
        target_ids: Tensor,
        style_images: Tensor | None = None,
        geometry_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        skips: list[Tensor] = []
        x = images
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.encoder_blocks) - 1:
                skips.append(x)
                x = self.pool(x)

        style_features = None
        if style_images is not None:
            style_features = style_images
            for index, block in enumerate(self.encoder_blocks):
                style_features = block(style_features)
                if index < len(self.encoder_blocks) - 1:
                    style_features = self.pool(style_features)

        prompt, style_embedding = self.task_prompt.compose(
            x,
            operation_ids,
            target_ids,
            style_features=style_features,
        )
        x, channel_gate, spatial_gate = self.modulator(x, prompt)

        for block, skip in zip(self.decoder_blocks, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))

        outputs = {
            "logits": self.output_head(x, prompt),
            "prompt": prompt,
            "channel_gate": channel_gate,
            "spatial_gate": spatial_gate,
        }
        if style_embedding is not None:
            outputs["style_embedding"] = style_embedding
        if geometry_masks is not None:
            if self.geometry_prototypes is None:
                raise ValueError("Geometry masks were provided to a model without prototypes")
            geometry_loss, sample_count = self.geometry_prototypes(
                x,
                geometry_masks,
                style_embeddings=style_embedding,
            )
            outputs["geometry_loss"] = geometry_loss
            outputs["geometry_sample_count"] = sample_count
        return outputs
