from __future__ import annotations

from pathlib import Path

from .audit import (
    analyze_manifest,
    write_audit_json,
    write_audit_markdown,
    write_shortlist_csv,
)
from .config import DataPipelineConfig
from .manifest import build_manifest, read_manifest


def run_data_pipeline(config: DataPipelineConfig, *, rebuild_manifest: bool = True) -> dict:
    if rebuild_manifest:
        records = build_manifest(
            dataset_root=config.dataset_root,
            classes_file=config.classes_file,
            images_root=config.images_root,
            split_files=config.split_files,
            output=config.manifest_path,
            label_base=config.label_base,
            workers=config.workers,
        )
    else:
        if not config.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {config.manifest_path}")
        records = read_manifest(config.manifest_path)
    audit = analyze_manifest(
        records,
        policy=config.shortlist_policy,
        near_duplicate_distance=config.near_duplicate_distance,
        near_duplicate_report_limit=config.near_duplicate_report_limit,
    )
    write_audit_json(audit, config.audit_json_path, config.manifest_path)
    write_audit_markdown(audit, config.audit_markdown_path)
    write_shortlist_csv(audit, config.shortlist_path)
    return {
        "manifest": str(Path(config.manifest_path).resolve()),
        "audit_json": str(Path(config.audit_json_path).resolve()),
        "audit_markdown": str(Path(config.audit_markdown_path).resolve()),
        "shortlist": str(Path(config.shortlist_path).resolve()),
        "records": audit["records"],
        "classes": audit["class_count"],
        "selected_classes": sum(row["provisional_selected"] for row in audit["classes"]),
    }
