from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CryoNuSegAnnotation:
    key: str
    relative_root: Path
    annotator: str
    round: int | None
    declared_instances: int
    visible_instances: int


CRYONUSEG_ANNOTATIONS = {
    annotation.key: annotation
    for annotation in (
        CryoNuSegAnnotation(
            key="ann1_round1",
            relative_root=Path("Annotator 1 (biologist)"),
            annotator="Annotator 1 (biologist)",
            round=1,
            declared_instances=7596,
            visible_instances=7592,
        ),
        CryoNuSegAnnotation(
            key="ann1_round2",
            relative_root=Path(
                "Annotator 1 (biologist second round of manual marks up)"
            )
            / "Annotator 1 (biologist second round of manual marks up)",
            annotator="Annotator 1 (biologist)",
            round=2,
            declared_instances=8044,
            visible_instances=8036,
        ),
        CryoNuSegAnnotation(
            key="ann2",
            relative_root=Path("Annotator 2 (bioinformatician)")
            / "Annotator 2 (bioinformatician)",
            annotator="Annotator 2 (bioinformatician)",
            round=None,
            declared_instances=8251,
            visible_instances=8242,
        ),
    )
}


CRYONUSEG_ORGANS = {
    "AdrenalGland": "adrenal_gland",
    "Larynx": "larynx",
    "LymphNodes": "lymph_node",
    "Mediastinum": "mediastinum",
    "Pancreas": "pancreas",
    "Pleura": "pleura",
    "Skin": "skin",
    "Testes": "testis",
    "Thymus": "thymus",
    "ThyroidGland": "thyroid_gland",
}


def cryonuseg_organ(sample_id: str) -> str:
    prefix = "Human_"
    if not sample_id.startswith(prefix) or "_" not in sample_id[len(prefix) :]:
        raise ValueError(f"Unexpected CryoNuSeg sample ID: {sample_id}")
    organ_code, image_number = sample_id[len(prefix) :].rsplit("_", maxsplit=1)
    if organ_code not in CRYONUSEG_ORGANS or not (
        len(image_number) == 2 and image_number.isdigit()
    ):
        raise ValueError(f"Unexpected CryoNuSeg sample ID: {sample_id}")
    return CRYONUSEG_ORGANS[organ_code]


def cryonuseg_annotation_root(source: Path, annotation_key: str) -> Path:
    try:
        annotation = CRYONUSEG_ANNOTATIONS[annotation_key]
    except KeyError as error:
        raise ValueError(f"Unknown CryoNuSeg annotation: {annotation_key}") from error
    return source / annotation.relative_root
