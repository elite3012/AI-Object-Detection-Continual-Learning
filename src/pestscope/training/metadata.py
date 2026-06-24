from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PestClass:
    ip102_id: int
    dataset_label: str
    canonical_name: str
    common_name_en: str
    common_name_vi: str
    stratum: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_class_review(path: Path) -> dict[int, PestClass]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = raw.get("classes", []) if isinstance(raw, dict) else []
    reviewed: dict[int, PestClass] = {}
    for item in classes:
        pest = PestClass(
            ip102_id=int(item["ip102_id"]),
            dataset_label=str(item["dataset_label"]),
            canonical_name=str(item["canonical_name"]),
            common_name_en=str(item["common_name_en"]),
            common_name_vi=str(item["common_name_vi"]),
            stratum=str(item["stratum"]),
        )
        reviewed[pest.ip102_id] = pest
    if not reviewed:
        raise ValueError(f"No reviewed classes found in {path}")
    return reviewed
