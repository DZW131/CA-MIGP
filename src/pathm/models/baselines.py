from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .network import ConvBlock


class UNetFeatures(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.output_channels = channels[0]
        self.encoder_blocks = nn.ModuleList(
            [
                ConvBlock(3, channels[0]),
                ConvBlock(channels[0], channels[1]),
                ConvBlock(channels[1], channels[2]),
                ConvBlock(channels[2], channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.decoder_blocks = nn.ModuleList(
            [
                ConvBlock(channels[3] + channels[2], channels[2]),
                ConvBlock(channels[2] + channels[1], channels[1]),
                ConvBlock(channels[1] + channels[0], channels[0]),
            ]
        )

    def forward(self, images: Tensor) -> Tensor:
        skips: list[Tensor] = []
        features = images
        for index, block in enumerate(self.encoder_blocks):
            features = block(features)
            if index < len(self.encoder_blocks) - 1:
                skips.append(features)
                features = self.pool(features)
        for block, skip in zip(self.decoder_blocks, reversed(skips), strict=True):
            features = F.interpolate(
                features, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = block(torch.cat((features, skip), dim=1))
        return features


class SpecialistUNet(nn.Module):
    """Single-output U-Net trained for one operation only."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.features = UNetFeatures(base_channels)
        self.output_head = nn.Conv2d(self.features.output_channels, 1, kernel_size=1)

    def forward(
        self,
        images: Tensor,
        operation_ids: Tensor,
        target_ids: Tensor,
    ) -> dict[str, Tensor]:
        del operation_ids, target_ids
        return {"logits": self.output_head(self.features(images))}


class MultiHeadUNet(nn.Module):
    """Shared U-Net with a static output head for each operation."""

    def __init__(self, base_channels: int = 32, num_operations: int = 2) -> None:
        super().__init__()
        self.num_operations = num_operations
        self.features = UNetFeatures(base_channels)
        self.output_heads = nn.ModuleList(
            nn.Conv2d(self.features.output_channels, 1, kernel_size=1)
            for _ in range(num_operations)
        )

    def forward(
        self,
        images: Tensor,
        operation_ids: Tensor,
        target_ids: Tensor,
    ) -> dict[str, Tensor]:
        del target_ids
        if torch.any((operation_ids < 0) | (operation_ids >= self.num_operations)):
            raise ValueError("operation_ids contain an unavailable output head")
        features = self.features(images)
        all_logits = torch.stack([head(features) for head in self.output_heads], dim=1)
        batch_indices = torch.arange(images.shape[0], device=images.device)
        return {"logits": all_logits[batch_indices, operation_ids]}
