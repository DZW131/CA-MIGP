from __future__ import annotations

import torch
import torch.nn.functional as F

from pathm.training.losses import (
    MixedTaskLoss,
    center_heatmap_focal_loss,
    soft_dice_loss,
    uncertainty_boundary_weights,
)


def test_center_heatmap_focal_loss_prefers_correct_peak() -> None:
    target = torch.zeros(1, 1, 9, 9)
    target[0, 0, 4, 4] = 1.0
    correct = torch.full_like(target, -5.0)
    correct[0, 0, 4, 4] = 5.0
    incorrect = -correct
    assert center_heatmap_focal_loss(correct, target).item() < center_heatmap_focal_loss(
        incorrect, target
    ).item()


def test_center_heatmap_focal_loss_is_finite_for_extreme_bfloat16_logits() -> None:
    target = torch.zeros(2, 1, 9, 9, dtype=torch.bfloat16)
    target[:, :, 4, 4] = 1.0
    logits = torch.tensor([-100.0, 100.0], dtype=torch.bfloat16).view(2, 1, 1, 1)
    logits = logits.expand_as(target)
    assert torch.isfinite(center_heatmap_focal_loss(logits, target)).all()


def _geometry_masks() -> torch.Tensor:
    masks = torch.zeros(2, 5, 4, 4, dtype=torch.bool)
    masks[:, 0, 0, :] = True
    masks[:, 1, 1, :] = True
    masks[:, 2, 2, :] = True
    masks[:, 4, 3, :] = True
    return masks


def test_uncertainty_alpha_zero_recovers_original_loss_exactly() -> None:
    logits = torch.randn(2, 1, 4, 4)
    targets = torch.randint(0, 2, logits.shape).float()
    operations = torch.tensor([1, 1])
    original = MixedTaskLoss()(logits, targets, operations)
    zero_alpha = MixedTaskLoss(
        uncertainty_boundary_alpha=0.0,
        uncertainty_boundary_mode="state",
    )(logits, targets, operations, geometry_masks=_geometry_masks())
    torch.testing.assert_close(original, zero_alpha, rtol=0.0, atol=0.0)


def test_uncertainty_weights_only_selected_boundary_states() -> None:
    logits = torch.linspace(-4.0, 4.0, 32).reshape(2, 1, 4, 4)
    weights = uncertainty_boundary_weights(
        logits,
        _geometry_masks(),
        alpha=3.0,
        mode="state",
    )
    assert torch.all(weights[:, :, 2, :] == 1.0)
    assert torch.all(weights[:, :, :2, :] > 1.0)
    assert torch.all(weights[:, :, 3, :] > 1.0)


def test_uncertain_boundary_pixel_receives_more_weight() -> None:
    logits = torch.tensor([[[[0.0, 8.0], [-8.0, 8.0]]]])
    masks = torch.zeros(1, 5, 2, 2, dtype=torch.bool)
    masks[0, 0, 0, :] = True
    weights = uncertainty_boundary_weights(
        logits,
        masks,
        alpha=3.0,
        mode="state",
    )
    assert weights[0, 0, 0, 0] > weights[0, 0, 0, 1] > 1.0
    assert torch.all(weights[0, 0, 1] == 1.0)


def test_global_and_state_normalization_are_distinct() -> None:
    logits = torch.tensor(
        [[[[0.0, 0.5], [4.0, 5.0]]], [[[1.0, 2.0], [6.0, 7.0]]]]
    )
    masks = torch.zeros(2, 5, 2, 2, dtype=torch.bool)
    masks[:, 0, 0, :] = True
    masks[:, 1, 1, :] = True
    global_weights = uncertainty_boundary_weights(
        logits, masks, alpha=3.0, mode="global"
    )
    state_weights = uncertainty_boundary_weights(
        logits, masks, alpha=3.0, mode="state"
    )
    assert not torch.allclose(global_weights, state_weights)
    expected_support = masks[:, (0, 1, 4)].any(dim=1, keepdim=True)
    assert torch.equal(global_weights > 1.0, expected_support)
    assert torch.equal(state_weights > 1.0, expected_support)


def test_uncertainty_weight_signal_is_detached_from_gradient() -> None:
    logits = torch.linspace(-2.0, 2.0, 32).reshape(2, 1, 4, 4).requires_grad_()
    targets = torch.randint(0, 2, logits.shape).float()
    operations = torch.tensor([1, 1])
    masks = _geometry_masks()

    actual = MixedTaskLoss(
        uncertainty_boundary_alpha=3.0,
        uncertainty_boundary_mode="state",
    )(logits, targets, operations, geometry_masks=masks)
    actual_gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]

    fixed_weights = uncertainty_boundary_weights(
        logits.detach(), masks, alpha=3.0, mode="state"
    )
    manual = (
        F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        * fixed_weights
    ).mean(dim=(1, 2, 3))
    manual = (manual + soft_dice_loss(logits, targets)).mean()
    manual_gradient = torch.autograd.grad(manual, logits)[0]

    torch.testing.assert_close(actual, manual)
    torch.testing.assert_close(actual_gradient, manual_gradient)


def test_uncertainty_loss_leaves_detection_samples_unchanged() -> None:
    logits = torch.randn(2, 1, 4, 4)
    targets = torch.zeros_like(logits)
    targets[:, :, 2, 2] = 1.0
    operations = torch.tensor([0, 0])
    expected = center_heatmap_focal_loss(logits, targets).mean()
    actual = MixedTaskLoss(
        uncertainty_boundary_alpha=3.0,
        uncertainty_boundary_mode="state",
    )(logits, targets, operations, geometry_masks=_geometry_masks())
    torch.testing.assert_close(actual, expected)


def test_uncertainty_weights_handle_empty_states_and_backpropagate() -> None:
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    targets = torch.randint(0, 2, logits.shape).float()
    masks = torch.zeros(2, 5, 4, 4, dtype=torch.bool)
    masks[0, 0, 1, 1] = True
    loss = MixedTaskLoss(
        uncertainty_boundary_alpha=3.0,
        uncertainty_boundary_mode="state",
    )(logits, targets, torch.tensor([1, 1]), geometry_masks=masks)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert not torch.allclose(
        loss.detach(),
        (
            F.binary_cross_entropy_with_logits(logits.detach(), targets)
            + MixedTaskLoss()(logits.detach(), targets, torch.tensor([1, 1]))
            - F.binary_cross_entropy_with_logits(logits.detach(), targets)
        ),
    )
