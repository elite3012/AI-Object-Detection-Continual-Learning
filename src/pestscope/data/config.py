from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .audit import ShortlistPolicy


@dataclass(frozen=True)
class DataPipelineConfig:
    dataset_root: Path
    classes_file: Path
    images_root: Path
    split_files: dict[str, Path]
    manifest_path: Path
    audit_json_path: Path
    audit_markdown_path: Path
    shortlist_path: Path
    label_base: str
    workers: int
    near_duplicate_distance: int
    near_duplicate_report_limit: int
    shortlist_policy: ShortlistPolicy


class ConfigError(ValueError):
    """Raised when the data-pipeline configuration is incomplete."""


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def load_data_config(path: Path, dataset_root_override: Path | None = None) -> DataPipelineConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    dataset = _mapping(root.get("dataset"), "dataset")
    outputs = _mapping(root.get("outputs"), "outputs")
    manifest = _mapping(root.get("manifest", {}), "manifest")
    audit = _mapping(root.get("audit", {}), "audit")
    shortlist = _mapping(root.get("shortlist", {}), "shortlist")
    split_values = _mapping(dataset.get("split_files"), "dataset.split_files")

    missing_splits = {"train", "val", "test"} - set(split_values)
    if missing_splits:
        raise ConfigError(f"Missing split files: {', '.join(sorted(missing_splits))}")

    dataset_root = dataset_root_override or Path(str(dataset.get("root", "data/raw/ip102")))
    return DataPipelineConfig(
        dataset_root=dataset_root,
        classes_file=Path(str(dataset.get("classes_file", "classes.txt"))),
        images_root=Path(str(dataset.get("images_root", "images"))),
        split_files={name: Path(str(value)) for name, value in split_values.items()},
        manifest_path=Path(str(outputs.get("manifest", "artifacts/data/ip102_manifest.csv"))),
        audit_json_path=Path(str(outputs.get("audit_json", "artifacts/data/ip102_audit.json"))),
        audit_markdown_path=Path(str(outputs.get("audit_markdown", "artifacts/data/ip102_eda.md"))),
        shortlist_path=Path(str(outputs.get("shortlist", "artifacts/data/ip102_shortlist.csv"))),
        label_base=str(manifest.get("label_base", "auto")),
        workers=int(manifest.get("workers", 4)),
        near_duplicate_distance=int(audit.get("near_duplicate_distance", 4)),
        near_duplicate_report_limit=int(audit.get("near_duplicate_report_limit", 500)),
        shortlist_policy=ShortlistPolicy(
            target_classes=int(shortlist.get("target_classes", 12)),
            min_train=int(shortlist.get("min_train", 100)),
            min_val=int(shortlist.get("min_val", 20)),
            min_test=int(shortlist.get("min_test", 20)),
            max_exact_cross_split_duplicate_records=int(
                shortlist.get("max_exact_cross_split_duplicate_records", 0)
            ),
            manual_class_ids=tuple(
                int(class_id) for class_id in shortlist.get("manual_class_ids", [])
            ),
        ),
    )
