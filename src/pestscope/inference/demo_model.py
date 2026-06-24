from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch

from pestscope.modeling import build_model, count_parameters
from pestscope.training.bundle import write_model_bundle
from pestscope.training.metadata import load_class_review
from pestscope.training.transforms import DEFAULT_MEAN, DEFAULT_STD


def ensure_demo_bundle(
    *,
    bundle_dir: Path,
    class_review_path: Path,
    image_size: int = 224,
) -> Path:
    if (bundle_dir / "metadata.json").is_file() and (bundle_dir / "model.pt").is_file():
        return bundle_dir

    reviewed = load_class_review(class_review_path)
    selected_ids = sorted(reviewed)
    torch.manual_seed(2026)
    model = build_model("pestnet_s", num_classes=len(selected_ids), width=8, dropout=0.0)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "demo-untrained",
        "demo_model": True,
        "warning": (
            "This fallback bundle is for API/UI smoke testing only. Train PestNet-S and "
            "point PESTSCOPE_MODEL_BUNDLE at a promoted bundle before reporting metrics."
        ),
        "dataset": {
            "root": None,
            "manifest": None,
            "manifest_sha256": None,
            "selected_class_ids": selected_ids,
            "train_records": 0,
            "val_records": 0,
        },
        "model": {
            "name": "pestnet_s",
            "width": 8,
            "dropout": 0.0,
            "num_classes": len(selected_ids),
            "parameter_count": count_parameters(model),
        },
        "preprocessing": {
            "image_size": image_size,
            "mean": list(DEFAULT_MEAN),
            "std": list(DEFAULT_STD),
        },
        "classes": [
            {"index": index, **reviewed[class_id].to_dict()}
            for index, class_id in enumerate(selected_ids)
        ],
        "training": {
            "seed": 2026,
            "epochs_run": 0,
            "device": "cpu",
            "class_strategy": "none",
        },
    }
    metrics = {
        "schema_version": 1,
        "run_id": "demo-untrained",
        "best_validation": None,
        "history": [],
        "warning": "No accuracy claim is attached to the fallback demo model.",
    }
    write_model_bundle(bundle_dir=bundle_dir, model=model, metadata=metadata, metrics=metrics)
    return bundle_dir
