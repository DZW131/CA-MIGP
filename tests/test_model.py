import torch

from pathm.models import (
    MultiHeadUNet,
    PathMUnified,
    SpecialistUNet,
    MultiGranularInstanceGeometryPrototypes,
    build_model,
)
from pathm.training import MixedTaskLoss


def test_mixed_task_forward_and_backward() -> None:
    model = PathMUnified(base_channels=8, prompt_dim=32, style_channels=8)
    images = torch.rand(2, 3, 64, 64)
    operations = torch.tensor([0, 1])
    targets = torch.tensor([0, 1])
    labels = torch.rand(2, 1, 64, 64)

    output = model(images, operations, targets)
    assert output["logits"].shape == labels.shape
    assert output["prompt"].shape == (2, 32)
    assert output["channel_gate"].shape == (2, 64, 1, 1)
    assert output["spatial_gate"].shape == (2, 1, 8, 8)

    loss = MixedTaskLoss()(output["logits"], labels, operations)
    loss.backward()
    assert torch.isfinite(loss)


def test_style_direction_removal_is_orthogonal() -> None:
    features = torch.randn(8, 16)
    directions = torch.randn(8, 16)
    orthogonal = MultiGranularInstanceGeometryPrototypes.remove_style_direction(
        features, directions
    )
    cosine_projection = (
        orthogonal * torch.nn.functional.normalize(directions, dim=1)
    ).sum(dim=1)
    assert torch.allclose(cosine_projection, torch.zeros_like(cosine_projection), atol=1e-5)


def test_geometry_prototypes_forward_backward_and_update() -> None:
    model = PathMUnified(
        base_channels=8,
        prompt_dim=32,
        style_channels=8,
        use_geometry_prototypes=True,
        geometry_num_prototypes=5,
        geometry_max_samples_per_prototype=16,
        geometry_prototype_weights=(1.0, 2.0, 1.0, 1.0, 1.0),
    )
    images = torch.rand(2, 3, 64, 64)
    operations = torch.tensor([0, 1])
    target_ids = torch.zeros(2, dtype=torch.long)
    labels = torch.rand(2, 1, 64, 64)
    geometry_masks = torch.zeros(2, 5, 64, 64, dtype=torch.bool)
    geometry_masks[1, 0, 8:16, 8:16] = True
    geometry_masks[1, 1, 8:16, 16:24] = True
    geometry_masks[1, 2, 16:24, 8:16] = True
    geometry_masks[1, 3, 16:24, 16:24] = True
    geometry_masks[1, 4, 24:32, 8:16] = True

    output = model(
        images,
        operations,
        target_ids,
        geometry_masks=geometry_masks,
    )
    assert output["style_embedding"].shape == (2, 32)
    assert torch.isfinite(output["geometry_loss"])
    assert output["geometry_sample_count"].item() == 80
    assert model.geometry_prototypes is not None
    assert torch.all(model.geometry_prototypes.prototype_counts > 0)

    task_loss = MixedTaskLoss()(output["logits"], labels, operations)
    (task_loss + 0.1 * output["geometry_loss"]).backward()
    assert model.geometry_prototypes.projection[0].weight.grad is not None


def test_geometry_prototypes_can_run_without_style_orthogonalization() -> None:
    model = PathMUnified(
        base_channels=8,
        prompt_dim=16,
        style_channels=4,
        use_style_prompt=False,
        use_geometry_prototypes=True,
        geometry_num_prototypes=3,
        geometry_style_direction_mode="none",
    )
    images = torch.rand(1, 3, 64, 64)
    masks = torch.zeros(1, 3, 64, 64, dtype=torch.bool)
    masks[0, 0, 4:8, 4:8] = True
    masks[0, 1, 8:12, 8:12] = True
    masks[0, 2, 12:16, 12:16] = True
    output = model(
        images,
        torch.ones(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        geometry_masks=masks,
    )
    assert torch.isfinite(output["geometry_loss"])


def test_model_ablation_switches_forward() -> None:
    images = torch.randn(2, 3, 64, 64)
    operation_ids = torch.tensor([0, 1])
    target_ids = torch.tensor([0, 2])
    for options in (
        {"use_style_prompt": False},
        {"factorized_task_prompt": False},
        {"use_channel_prompt": False},
        {"use_spatial_prompt": False},
        {"use_channel_prompt": False, "use_spatial_prompt": False},
    ):
        model = PathMUnified(base_channels=8, prompt_dim=16, style_channels=4, **options)
        outputs = model(images, operation_ids, target_ids)
        assert outputs["logits"].shape == (2, 1, 64, 64)
        assert torch.isfinite(outputs["logits"]).all()


def test_style_controls_are_parameter_matched() -> None:
    models = [
        PathMUnified(
            base_channels=8,
            prompt_dim=16,
            style_channels=4,
            style_prompt_mode=mode,
        )
        for mode in ("gram", "diagonal", "constant")
    ]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts[0] == counts[1] == counts[2]


def test_constant_style_is_image_independent() -> None:
    model = PathMUnified(
        base_channels=8,
        prompt_dim=16,
        style_channels=4,
        style_prompt_mode="constant",
    ).eval()
    images = torch.stack((torch.zeros(3, 64, 64), torch.ones(3, 64, 64)))
    operation_ids = torch.zeros(2, dtype=torch.long)
    target_ids = torch.zeros(2, dtype=torch.long)
    prompts = model(images, operation_ids, target_ids)["prompt"]
    assert torch.allclose(prompts[0], prompts[1])


def test_mismatched_style_source_changes_prompt() -> None:
    model = PathMUnified(base_channels=8, prompt_dim=16, style_channels=4).eval()
    images = torch.stack((torch.zeros(3, 64, 64), torch.ones(3, 64, 64)))
    operation_ids = torch.zeros(2, dtype=torch.long)
    target_ids = torch.zeros(2, dtype=torch.long)
    matched = model(images, operation_ids, target_ids)["prompt"]
    mismatched = model(
        images,
        operation_ids,
        target_ids,
        style_images=images.flip(0),
    )["prompt"]
    assert torch.allclose(matched[0], mismatched[1])
    assert torch.allclose(matched[1], mismatched[0])


def test_baseline_models_forward() -> None:
    images = torch.randn(2, 3, 64, 64)
    operation_ids = torch.tensor([0, 1])
    target_ids = torch.tensor([0, 0])
    for model in (MultiHeadUNet(base_channels=8), SpecialistUNet(base_channels=8)):
        assert model(images, operation_ids, target_ids)["logits"].shape == (2, 1, 64, 64)


def test_model_factory() -> None:
    assert isinstance(build_model({"architecture": "multihead_unet", "base_channels": 8}), MultiHeadUNet)
    assert isinstance(build_model({"architecture": "specialist_unet", "base_channels": 8}), SpecialistUNet)
    assert isinstance(build_model({"architecture": "prompt_unified", "base_channels": 8}), PathMUnified)
