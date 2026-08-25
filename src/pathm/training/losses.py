from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


UNCERTAINTY_BOUNDARY_MODES = {"none", "global", "state"}


def soft_dice_loss(logits: Tensor, targets: Tensor, epsilon: float = 1e-6) -> Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon))


def center_heatmap_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    epsilon: float = 1e-4,
) -> Tensor:
    """CenterNet-style focal loss normalized by the number of object centers."""
    probabilities = torch.sigmoid(logits.float()).clamp(epsilon, 1.0 - epsilon)
    float_targets = targets.float()
    positive_mask = float_targets.eq(1.0)
    negative_mask = float_targets.lt(1.0)
    negative_weights = (1.0 - float_targets).pow(beta)
    positive_loss = (
        torch.log(probabilities) * (1.0 - probabilities).pow(alpha) * positive_mask
    ).sum(dim=(1, 2, 3))
    negative_loss = (
        torch.log(1.0 - probabilities)
        * probabilities.pow(alpha)
        * negative_weights
        * negative_mask
    ).sum(dim=(1, 2, 3))
    positive_count = positive_mask.sum(dim=(1, 2, 3)).clamp_min(1)
    return -(positive_loss + negative_loss) / positive_count


def uncertainty_boundary_weights(
    logits: Tensor,
    geometry_masks: Tensor,
    *,
    alpha: float,
    mode: str,
    state_indices: tuple[int, ...] = (0, 1, 4),
    epsilon: float = 1e-6,
) -> Tensor:
    """Build detached entropy weights over selected boundary geometry states."""
    if alpha < 0.0:
        raise ValueError("uncertainty boundary alpha must be non-negative")
    if mode not in UNCERTAINTY_BOUNDARY_MODES:
        raise ValueError(f"Unknown uncertainty boundary mode: {mode}")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B, 1, H, W]")
    if geometry_masks.ndim != 4:
        raise ValueError("geometry_masks must have shape [B, P, H, W]")
    if geometry_masks.shape[0] != logits.shape[0] or geometry_masks.shape[-2:] != logits.shape[-2:]:
        raise ValueError("geometry masks and logits must have matching batch/spatial shapes")
    if not state_indices:
        raise ValueError("state_indices must not be empty")
    if any(index < 0 or index >= geometry_masks.shape[1] for index in state_indices):
        raise ValueError("state_indices contain an unavailable geometry state")

    weights = torch.ones_like(logits, dtype=torch.float32)
    if alpha == 0.0 or mode == "none":
        return weights.to(logits.dtype)

    probabilities = torch.sigmoid(logits.detach().float()).clamp(epsilon, 1.0 - epsilon)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
    masks = geometry_masks[:, state_indices].bool()

    def standardized_emphasis(mask: Tensor) -> Tensor:
        selected = entropy[:, 0][mask]
        if selected.numel() == 0:
            return selected
        z_score = (selected - selected.mean()) / (selected.std(unbiased=False) + epsilon)
        return 1.0 + alpha * torch.sigmoid(z_score)

    if mode == "global":
        boundary_mask = masks.any(dim=1)
        weights[:, 0][boundary_mask] = standardized_emphasis(boundary_mask)
    else:
        for state_offset in range(masks.shape[1]):
            state_mask = masks[:, state_offset]
            weights[:, 0][state_mask] = standardized_emphasis(state_mask)
    return weights.to(logits.dtype)


class MixedTaskLoss(nn.Module):
    """Use focal heatmap loss for detection and BCE+Dice for segmentation."""

    DETECTION_ID = 0
    SEGMENTATION_ID = 1

    def __init__(
        self,
        uncertainty_boundary_alpha: float = 0.0,
        uncertainty_boundary_mode: str = "none",
        uncertainty_boundary_state_indices: tuple[int, ...] = (0, 1, 4),
    ) -> None:
        super().__init__()
        if uncertainty_boundary_alpha < 0.0:
            raise ValueError("uncertainty boundary alpha must be non-negative")
        if uncertainty_boundary_mode not in UNCERTAINTY_BOUNDARY_MODES:
            raise ValueError(
                f"Unknown uncertainty boundary mode: {uncertainty_boundary_mode}"
            )
        if uncertainty_boundary_alpha > 0.0 and uncertainty_boundary_mode == "none":
            raise ValueError("A positive uncertainty alpha requires global or state mode")
        self.uncertainty_boundary_alpha = uncertainty_boundary_alpha
        self.uncertainty_boundary_mode = uncertainty_boundary_mode
        self.uncertainty_boundary_state_indices = uncertainty_boundary_state_indices

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        operation_ids: Tensor,
        geometry_masks: Tensor | None = None,
    ) -> Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"Logit shape {logits.shape} does not match target {targets.shape}")
        detection = center_heatmap_focal_loss(logits, targets)
        segmentation_bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        if self.uncertainty_boundary_alpha > 0.0:
            if geometry_masks is None:
                raise ValueError("Uncertainty boundary loss requires geometry masks")
            segmentation_bce = segmentation_bce * uncertainty_boundary_weights(
                logits,
                geometry_masks,
                alpha=self.uncertainty_boundary_alpha,
                mode=self.uncertainty_boundary_mode,
                state_indices=self.uncertainty_boundary_state_indices,
            )
        segmentation = segmentation_bce.mean(dim=(1, 2, 3)) + soft_dice_loss(
            logits, targets
        )
        per_sample = torch.where(
            operation_ids == self.DETECTION_ID,
            detection,
            segmentation,
        )
        return per_sample.mean()
