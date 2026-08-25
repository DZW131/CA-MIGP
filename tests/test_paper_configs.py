from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 23, 42)


def load_config(stem: str, seed: int) -> dict[str, object]:
    path = ROOT / "configs" / f"{stem}_seed{seed}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def without_identity(config: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(config)
    result.pop("run_name")
    result["training"].pop("seed")  # type: ignore[union-attr]
    return result


def test_each_paper_family_is_seed_paired() -> None:
    families = (
        "pannuke_ablation_no_style",
        "pannuke_full",
        "pannuke_no_style_geometry",
        "pannuke_geometry_no_orth",
        "pannuke_camigp_no_contact",
        "pannuke_gap_control",
        "pannuke_somigp",
        "pannuke_geometry_no_contact",
    )
    for family in families:
        configs = [without_identity(load_config(family, seed)) for seed in SEEDS]
        assert configs[0] == configs[1] == configs[2]


def test_factorial_cells_change_only_declared_components() -> None:
    base = load_config("pannuke_ablation_no_style", 17)
    gram = load_config("pannuke_full", 17)
    geometry = load_config("pannuke_no_style_geometry", 17)
    complete = load_config("pannuke_geometry_no_orth", 17)

    assert base["model"]["use_style_prompt"] is False  # type: ignore[index]
    assert gram["model"]["use_style_prompt"] is True  # type: ignore[index]
    assert "use_geometry_prototypes" not in base["model"]  # type: ignore[operator]
    assert "use_geometry_prototypes" not in gram["model"]  # type: ignore[operator]
    assert geometry["model"]["use_style_prompt"] is False  # type: ignore[index]
    assert complete["model"]["use_style_prompt"] is True  # type: ignore[index]
    assert geometry["model"]["use_geometry_prototypes"] is True  # type: ignore[index]
    assert complete["model"]["use_geometry_prototypes"] is True  # type: ignore[index]
    assert geometry["model"]["geometry_style_direction_mode"] == "none"  # type: ignore[index]
    assert complete["model"]["geometry_style_direction_mode"] == "none"  # type: ignore[index]


def test_matched_no_contact_changes_only_contact_package() -> None:
    complete = load_config("pannuke_geometry_no_orth", 17)
    control = load_config("pannuke_camigp_no_contact", 17)
    control["run_name"] = complete["run_name"]
    control["data"]["geometry_target_mode"] = "instance"  # type: ignore[index]
    control["model"]["geometry_num_prototypes"] = 5  # type: ignore[index]
    control["model"]["geometry_prototype_weights"] = [1.0, 2.0, 1.0, 1.0, 1.0]  # type: ignore[index]
    assert control == complete


def test_public_configs_do_not_contain_machine_paths() -> None:
    for path in (ROOT / "configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8").lower()
        assert "/home/" not in text
        assert "10.15." not in text
        assert "duyanhong" not in text
