from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


OPERATIONS = {"detection", "segmentation"}
TARGETS = {"nucleus", "gland", "tissue", "tumor", "other"}
SPLITS = {"train", "val", "test", "ood_test"}


@dataclass(frozen=True)
class ManifestRecord:
    image: str
    label: str
    operation: str
    target: str
    domain: str
    patient_id: str
    split: str
    ignore: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, str | None]) -> "ManifestRecord":
        required = {
            "image",
            "label",
            "operation",
            "target",
            "domain",
            "patient_id",
            "split",
        }
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Missing manifest fields: {sorted(missing)}")
        record = cls(
            **{key: value[key] for key in required},  # type: ignore[arg-type]
            ignore=value.get("ignore"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.operation not in OPERATIONS:
            raise ValueError(f"Unknown operation: {self.operation}")
        if self.target not in TARGETS:
            raise ValueError(f"Unknown target: {self.target}")
        if self.split not in SPLITS:
            raise ValueError(f"Unknown split: {self.split}")
        if not self.domain or not self.patient_id:
            raise ValueError("domain and patient_id must be non-empty")


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(ManifestRecord.from_dict(json.loads(line)))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
    return records


def validate_manifest(records: list[ManifestRecord], check_paths: bool = True) -> None:
    patient_splits: dict[tuple[str, str], set[str]] = {}
    for record in records:
        record.validate()
        if check_paths:
            for value in (record.image, record.label, record.ignore):
                if value is None:
                    continue
                if not Path(value).exists():
                    raise FileNotFoundError(value)
        patient_splits.setdefault((record.domain, record.patient_id), set()).add(record.split)
    leakage = {key: splits for key, splits in patient_splits.items() if len(splits) > 1}
    if leakage:
        preview = list(leakage.items())[:5]
        raise ValueError(f"Patient/WSI leakage across splits: {preview}")


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
