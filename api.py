from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.inference.config import InferenceSettings  # noqa: E402
from pestscope.inference.demo_model import ensure_demo_bundle  # noqa: E402
from pestscope.inference.examples import (  # noqa: E402
    DemoExample,
    example_image_bytes,
    load_demo_examples,
)
from pestscope.inference.reviews import ReviewStore  # noqa: E402
from pestscope.inference.service import (  # noqa: E402
    InferenceService,
    PredictionError,
    image_from_upload,
)


class ReviewRequest(BaseModel):
    prediction_id: str = Field(..., min_length=8, max_length=80)
    decision: str = Field(..., min_length=3, max_length=32)
    predicted_class_id: int | None = None
    corrected_class_id: int | None = None
    note: str | None = Field(default=None, max_length=500)
    image_consent: bool = False


def _project_path(value: object | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _short_hash(value: object | None, length: int = 12) -> str | None:
    if not value:
        return None
    return str(value)[:length]


def _metric_row(row: dict) -> dict:
    keys = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_macro_f1",
        "val_macro_f1",
        "train_top1_accuracy",
        "val_top1_accuracy",
    ]
    cleaned = {}
    for key in keys:
        raw = row.get(key)
        cleaned[key] = _safe_int(raw) if key == "epoch" else _safe_float(raw)
    return cleaned


def _class_distribution(metadata: dict, metrics: dict) -> list[dict]:
    dataset = metadata.get("dataset") or {}
    classes = sorted(metadata.get("classes") or [], key=lambda item: int(item.get("index", 0)))
    selected_ids = {
        int(class_id)
        for class_id in dataset.get("selected_class_ids", [])
        if _safe_int(class_id) is not None
    }
    if not selected_ids:
        selected_ids = {
            int(item.get("ip102_id"))
            for item in classes
            if _safe_int(item.get("ip102_id")) is not None
        }
    counts = {class_id: {"train": 0, "val": 0, "test": 0} for class_id in selected_ids}
    shortlist = {}

    manifest_path = _project_path(dataset.get("manifest"))
    if manifest_path and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                class_id = _safe_int(row.get("class_id"))
                split = str(row.get("split") or "").strip().lower()
                if class_id in counts and split in counts[class_id]:
                    counts[class_id][split] += 1

    shortlist_path = PROJECT_ROOT / "artifacts" / "data" / "ip102_shortlist.csv"
    if shortlist_path.is_file():
        with shortlist_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                class_id = _safe_int(row.get("class_id"))
                if class_id is None:
                    continue
                shortlist[class_id] = {
                    "stratum": row.get("stratum"),
                    "train": _safe_int(row.get("train")) or 0,
                    "val": _safe_int(row.get("val")) or 0,
                    "test": _safe_int(row.get("test")) or 0,
                    "selection_score": _safe_float(row.get("selection_score")),
                    "near_duplicates": _safe_int(row.get("near_cross_split_duplicate_pairs")) or 0,
                    "external_validation": row.get("external_validation"),
                }

    for class_id, split_counts in counts.items():
        if any(split_counts.values()):
            continue
        if class_id in shortlist:
            split_counts.update(
                {
                    "train": shortlist[class_id]["train"],
                    "val": shortlist[class_id]["val"],
                    "test": shortlist[class_id]["test"],
                }
            )

    support_by_index = {
        int(row.get("index")): _safe_int(row.get("support")) or 0
        for row in (metrics.get("best_validation") or {}).get("per_class", [])
        if _safe_int(row.get("index")) is not None
    }
    distribution = []
    for item in classes:
        class_id = _safe_int(item.get("ip102_id"))
        index = _safe_int(item.get("index"))
        if class_id is None:
            continue
        split_counts = counts.get(class_id, {"train": 0, "val": 0, "test": 0})
        if not split_counts["val"] and index in support_by_index:
            split_counts = {**split_counts, "val": support_by_index[index]}
        extra = shortlist.get(class_id, {})
        distribution.append(
            {
                "index": index,
                "class_id": class_id,
                "common_name_en": item.get("common_name_en"),
                "dataset_label": item.get("dataset_label"),
                "scientific_name": item.get("canonical_name"),
                "stratum": item.get("stratum") or extra.get("stratum"),
                "train": split_counts["train"],
                "val": split_counts["val"],
                "test": split_counts["test"],
                "total": split_counts["train"] + split_counts["val"] + split_counts["test"],
                "selection_score": extra.get("selection_score"),
                "near_duplicates": extra.get("near_duplicates"),
                "external_validation": extra.get("external_validation"),
            }
        )
    return distribution


def _confusion_pairs(matrix: list, classes: list[dict], limit: int = 6) -> list[dict]:
    labels = {int(item.get("index", 0)): item for item in classes}
    pairs = []
    for actual_index, row in enumerate(matrix or []):
        for predicted_index, count in enumerate(row or []):
            count = _safe_int(count) or 0
            if actual_index == predicted_index or count <= 0:
                continue
            actual = labels.get(actual_index, {})
            predicted = labels.get(predicted_index, {})
            pairs.append(
                {
                    "actual_index": actual_index,
                    "predicted_index": predicted_index,
                    "actual": actual.get("common_name_en")
                    or actual.get("dataset_label")
                    or str(actual_index),
                    "predicted": predicted.get("common_name_en")
                    or predicted.get("dataset_label")
                    or str(predicted_index),
                    "count": count,
                }
            )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:limit]


def _class_label(item: dict | None, fallback: object = "Unknown") -> str:
    if not item:
        return str(fallback)
    return str(item.get("common_name_en") or item.get("dataset_label") or fallback)


def _failure_reason(
    actual: dict | None, predicted: dict | None, *, recall: float | None = None
) -> str:
    actual_name = _class_label(actual).lower()
    predicted_name = _class_label(predicted).lower()
    actual_genus = str((actual or {}).get("canonical_name") or "").split(" ")[0].lower()
    predicted_genus = str((predicted or {}).get("canonical_name") or "").split(" ")[0].lower()
    if actual_genus and actual_genus == predicted_genus:
        return (
            "Same genus or close morphology; local texture is not enough to separate the classes."
        )
    if any(
        token in actual_name and token in predicted_name
        for token in ("worm", "whitefly", "mite", "borer")
    ):
        return "Shared body texture or silhouette makes the first-stage features overlap."
    if actual and predicted and actual.get("stratum") == predicted.get("stratum"):
        return (
            "Classes sit in the same difficulty stratum, so the model needs harder "
            "pairwise examples."
        )
    if recall is not None and recall < 0.4:
        return "Low recall suggests the model has not learned enough stable cues for this class."
    return "Background, scale, or crop context may dominate the insect-specific evidence."


def _failure_action(actual: dict | None, predicted: dict | None, distribution: dict | None) -> str:
    train_count = _safe_int((distribution or {}).get("train")) or 0
    actual_label = _class_label(actual).lower()
    predicted_label = _class_label(predicted).lower()
    if train_count and train_count < 260:
        return (
            "Add targeted images for this class first, then keep a balanced validation "
            "slice to verify recall."
        )
    if predicted:
        return (
            f"Create a hard-negative mini-set: {actual_label} versus {predicted_label}, "
            "then retrain with class-balanced sampling and stronger crop/background augmentation."
        )
    return (
        "Store sample-level validation errors, then retrain on the most frequent "
        "false-negative patterns."
    )


def _weakest_classes(
    metrics: dict, classes: list[dict], distribution: list[dict], matrix: list
) -> list[dict]:
    classes_by_index = {int(item.get("index", 0)): item for item in classes}
    distribution_by_index = {
        int(item.get("index")): item
        for item in distribution
        if _safe_int(item.get("index")) is not None
    }
    rows = []
    for row in (metrics.get("best_validation") or {}).get("per_class", []):
        index = _safe_int(row.get("index"))
        if index is None:
            continue
        actual = classes_by_index.get(index)
        confusion_row = matrix[index] if index < len(matrix) else []
        predicted_index = None
        predicted_count = 0
        for candidate_index, count in enumerate(confusion_row):
            count = _safe_int(count) or 0
            if candidate_index == index or count <= predicted_count:
                continue
            predicted_index = candidate_index
            predicted_count = count
        predicted = classes_by_index.get(predicted_index) if predicted_index is not None else None
        recall = _safe_float(row.get("recall"))
        precision = _safe_float(row.get("precision"))
        f1 = _safe_float(row.get("f1"))
        support = _safe_int(row.get("support")) or sum(
            _safe_int(item) or 0 for item in confusion_row
        )
        rows.append(
            {
                "index": index,
                "class_id": actual.get("ip102_id") if actual else None,
                "name": _class_label(actual, index),
                "scientific_name": (actual or {}).get("canonical_name"),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "most_confused_with": _class_label(predicted, predicted_index)
                if predicted
                else None,
                "confused_count": predicted_count,
                "why": _failure_reason(actual, predicted, recall=recall),
                "fix": _failure_action(actual, predicted, distribution_by_index.get(index)),
            }
        )
    return sorted(rows, key=lambda item: (item["f1"] if item["f1"] is not None else 1.0))[:4]


def _confusion_drivers(matrix: list, classes: list[dict], limit: int = 6) -> list[dict]:
    classes_by_index = {int(item.get("index", 0)): item for item in classes}
    drivers = []
    for actual_index, row in enumerate(matrix or []):
        support = sum(_safe_int(item) or 0 for item in row)
        if support <= 0:
            continue
        for predicted_index, count in enumerate(row or []):
            count = _safe_int(count) or 0
            if actual_index == predicted_index or count <= 0:
                continue
            actual = classes_by_index.get(actual_index)
            predicted = classes_by_index.get(predicted_index)
            drivers.append(
                {
                    "actual_index": actual_index,
                    "predicted_index": predicted_index,
                    "actual": _class_label(actual, actual_index),
                    "predicted": _class_label(predicted, predicted_index),
                    "count": count,
                    "share_of_actual": count / support,
                    "why": _failure_reason(actual, predicted),
                    "fix": _failure_action(actual, predicted, None),
                }
            )
    return sorted(drivers, key=lambda item: item["count"], reverse=True)[:limit]


def _external_benchmark_records() -> list[dict]:
    payload = _read_json(PROJECT_ROOT / "artifacts" / "external_benchmark" / "evaluation.json")
    return payload.get("records") or []


def _external_failure_cases(classes: list[dict], limit: int = 4) -> list[dict]:
    classes_by_id = {
        int(item.get("ip102_id")): item
        for item in classes
        if _safe_int(item.get("ip102_id")) is not None
    }
    cases = []
    for record in _external_benchmark_records():
        if not record.get("evaluated") or record.get("top1_correct") is True:
            continue
        class_id = _safe_int(record.get("class_id"))
        predicted_class_id = _safe_int(record.get("predicted_class_id"))
        image_path = _project_path(record.get("image_path"))
        if class_id is None or not image_path or not image_path.is_file():
            continue
        actual = classes_by_id.get(class_id)
        predicted = classes_by_id.get(predicted_class_id)
        if record.get("top3_correct") is False:
            why = (
                "The correct class is missing from top-3, so this is a representation "
                "gap, not only a threshold issue."
            )
        elif record.get("decision") == "unsupported":
            why = (
                "The gate rejects the image, but the ranking still points to a visually "
                "nearby class."
            )
        else:
            why = "The model accepts a wrong top-1 result, so this pair needs hard-negative review."
        cases.append(
            {
                "class_id": class_id,
                "actual": _class_label(actual, record.get("canonical_name") or class_id),
                "scientific_name": (actual or {}).get("canonical_name")
                or record.get("canonical_name"),
                "predicted_class_id": predicted_class_id,
                "predicted": _class_label(predicted, predicted_class_id),
                "decision": record.get("decision"),
                "confidence": _safe_float(record.get("confidence")),
                "top3_correct": bool(record.get("top3_correct")),
                "provider": record.get("provider"),
                "license": record.get("license"),
                "image_url": f"/api/v1/failure-cases/{class_id}/image",
                "why": why,
                "fix": _failure_action(actual, predicted, None),
            }
        )
    return sorted(cases, key=lambda item: item["confidence"] or 0, reverse=True)[:limit]


def _failure_analysis(metadata: dict, metrics: dict, distribution: list[dict]) -> dict:
    classes = sorted(metadata.get("classes") or [], key=lambda item: int(item.get("index", 0)))
    best = metrics.get("best_validation") or {}
    matrix = best.get("confusion_matrix") or []
    weakest = _weakest_classes(metrics, classes, distribution, matrix)
    drivers = _confusion_drivers(matrix, classes)
    hard_cases = _external_failure_cases(classes)
    biggest_driver = drivers[0] if drivers else {}
    weakest_class = weakest[0] if weakest else {}
    return {
        "summary": {
            "primary_failure": (
                f"{weakest_class.get('name')} has the weakest F1"
                if weakest_class
                else "No per-class failures available"
            ),
            "largest_confusion": (
                f"{biggest_driver.get('actual')} -> {biggest_driver.get('predicted')}"
                if biggest_driver
                else "No confusion pair available"
            ),
            "hard_case_count": len(hard_cases),
            "sample_level_note": (
                "Validation artifacts store aggregate confusion and per-class metrics. "
                "External benchmark records provide the image-level hard cases shown here."
            ),
        },
        "weakest_classes": weakest,
        "confusion_drivers": drivers,
        "hard_cases": hard_cases,
        "root_causes": [
            {
                "title": "Low recall on visually variable classes",
                "evidence": (
                    "Beet armyworm and tobacco cutworm have low recall while being "
                    "frequently redirected to other moth/larval classes."
                ),
                "action": (
                    "Build a hard-negative validation set for the top confused pairs "
                    "before changing architecture."
                ),
            },
            {
                "title": "Tail classes have fewer stable examples",
                "evidence": (
                    "Mango shoot borer, citrus leafminer, and green mirid bug have "
                    "fewer selected training images and weaker F1."
                ),
                "action": (
                    "Add targeted data for tail classes, then retrain with "
                    "class-balanced sampling or focal loss ablation."
                ),
            },
            {
                "title": "Background and crop context can dominate",
                "evidence": (
                    "Some external hard cases are unsupported or misranked even when "
                    "the target is in top-3."
                ),
                "action": (
                    "Add insect-centered crop checks, background randomization, and "
                    "store sample-level top-k errors in the next evaluation run."
                ),
            },
        ],
        "improvement_steps": [
            (
                "Export sample-level validation predictions with image_id, path, "
                "target, top-k, confidence, and decision."
            ),
            (
                "Prioritize the largest confusion pairs as hard negatives, not only "
                "the lowest-F1 classes."
            ),
            (
                "Retrain one controlled ablation with stronger crop/background "
                "augmentation and compare macro-F1 plus accepted precision."
            ),
            "Promote only if the failure cards improve, not just if aggregate top-1 increases.",
        ],
    }


def _display_path(path: Path | str | None) -> str:
    if path is None:
        return "n/a"
    candidate = Path(path)
    try:
        candidate = candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    return str(candidate).replace("/", "\\")


def _artifact_status(label: str, relative_path: Path | str, kind: str, description: str) -> dict:
    path = _project_path(relative_path)
    exists = False
    if path:
        exists = path.is_dir() if kind == "directory" else path.is_file()
    return {
        "label": label,
        "path": _display_path(relative_path),
        "kind": kind,
        "exists": exists,
        "description": description,
    }


def _reproducibility(metadata: dict) -> dict:
    dataset = metadata.get("dataset") or {}
    model = metadata.get("model") or {}
    training = metadata.get("training") or {}
    artifact = metadata.get("artifact") or {}
    calibration = metadata.get("calibration") or {}

    model_name = str(model.get("name") or "pestnet_s")
    config_path = Path("configs") / "train" / f"{model_name}.yaml"
    if not (_project_path(config_path) or Path()).is_file():
        config_path = Path("configs") / "train" / "pestnet_s.yaml"

    bundle_dir = Path("artifacts") / "models" / f"{model_name}_latest"
    run_id = str(metadata.get("run_id") or "latest")
    run_dir = Path("artifacts") / "runs" / model_name / run_id
    eval_output = (
        calibration.get("output_path")
        or Path("artifacts") / "evaluation" / f"{model_name}_latest_eval.json"
    )
    eval_path = Path(str(eval_output))
    manifest_path = Path(str(dataset.get("manifest") or "artifacts/data/ip102_manifest.csv"))

    train_command = (
        f"python scripts\\train_pestnet.py --config {_display_path(config_path)} "
        "--device cuda --progress"
    )
    eval_command = (
        f"python scripts\\evaluate_pestnet_bundle.py --config {_display_path(config_path)} "
        f"--bundle-dir {_display_path(bundle_dir)} --split val --device cpu "
        f"--output {_display_path(eval_path)} --write-thresholds"
    )

    artifacts = [
        _artifact_status(
            "Training config",
            config_path,
            "file",
            "Single source for seed, classes, model width, optimizer, and output folders.",
        ),
        _artifact_status(
            "Dataset manifest",
            manifest_path,
            "file",
            "Frozen train/validation/test image list; evaluation does not reshuffle it.",
        ),
        _artifact_status(
            "Run folder",
            run_dir,
            "directory",
            "Training run output directory for logs and history.",
        ),
        _artifact_status(
            "History CSV",
            run_dir / "history.csv",
            "file",
            "Epoch-by-epoch loss and validation metrics.",
        ),
        _artifact_status(
            "Model bundle", bundle_dir, "directory", "Deployable bundle loaded by the API."
        ),
        _artifact_status(
            "Checkpoint weights",
            bundle_dir / str(artifact.get("model_file") or "model.pt"),
            "file",
            "Trained CNN weights used by the running app.",
        ),
        _artifact_status(
            "Bundle metadata",
            bundle_dir / "metadata.json",
            "file",
            "Run id, git commit, seed, dataset hash, and checkpoint hash.",
        ),
        _artifact_status(
            "Validation metrics",
            bundle_dir / str(artifact.get("metrics_file") or "metrics.json"),
            "file",
            "Best validation scores, per-class metrics, curves, and confusion matrix.",
        ),
        _artifact_status(
            "Evaluation report",
            eval_path,
            "file",
            "Calibration and threshold report produced after validation scoring.",
        ),
    ]

    return {
        "seed": training.get("seed"),
        "config_path": _display_path(config_path),
        "config_exists": bool((_project_path(config_path) or Path()).is_file()),
        "commands": {
            "train": train_command,
            "evaluate": eval_command,
        },
        "artifacts": artifacts,
        "run_contract": [
            (
                "Train reads the YAML config and writes a new timestamped folder "
                "under artifacts\\runs."
            ),
            (
                "Bundle metadata records the seed, selected classes, manifest hash, "
                "git commit, and checkpoint hash."
            ),
            (
                "Evaluation loads the bundle, scores the validation split, writes the "
                "report, then updates decision thresholds."
            ),
        ],
        "config_summary": {
            "selected_classes": len(dataset.get("selected_class_ids") or []),
            "image_size": (metadata.get("preprocessing") or {}).get("image_size"),
            "epochs": training.get("epochs_requested") or training.get("epochs_run"),
            "batch_size": training.get("batch_size"),
            "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"),
            "class_strategy": training.get("class_strategy"),
        },
    }


def experiment_evidence_payload(metadata: dict) -> dict:
    dataset = metadata.get("dataset") or {}
    model = metadata.get("model") or {}
    training = metadata.get("training") or {}
    artifact = metadata.get("artifact") or {}
    metrics = metadata.get("metrics") or {}
    best = metrics.get("best_validation") or {}
    classes = sorted(metadata.get("classes") or [], key=lambda item: int(item.get("index", 0)))
    audit = _read_json(PROJECT_ROOT / "artifacts" / "data" / "ip102_audit.json")
    augmentation = metadata.get("augmentation") or {
        "crop_scale": [0.82, 1.0],
        "hflip_probability": 0.5,
        "rotation_degrees": 10.0,
        "color_jitter": 0.12,
        "source": "training config default; not stored in this bundle metadata",
    }
    matrix = best.get("confusion_matrix") or []

    distribution = _class_distribution(metadata, metrics)
    return {
        "run": {
            "run_id": metadata.get("run_id"),
            "created_at": metadata.get("created_at"),
            "model_name": model.get("name"),
            "checkpoint_file": artifact.get("model_file"),
            "checkpoint_sha256": artifact.get("model_sha256"),
            "checkpoint_sha256_short": _short_hash(artifact.get("model_sha256")),
            "metrics_file": artifact.get("metrics_file"),
            "git_commit": (metadata.get("git") or {}).get("commit"),
            "git_commit_short": _short_hash((metadata.get("git") or {}).get("commit")),
            "git_dirty": bool((metadata.get("git") or {}).get("dirty", False)),
            "demo_model": bool(metadata.get("demo_model", False)),
        },
        "split": {
            "strategy": (
                "IP102 official train/val/test split; validation is used for model "
                "selection and calibration."
            ),
            "manifest_path": dataset.get("manifest"),
            "manifest_sha256": dataset.get("manifest_sha256"),
            "manifest_sha256_short": _short_hash(dataset.get("manifest_sha256")),
            "train_records": dataset.get("train_records"),
            "val_records": dataset.get("val_records"),
            "selected_class_ids": dataset.get("selected_class_ids") or [],
            "full_dataset": {
                "records": audit.get("records"),
                "class_count": audit.get("class_count"),
                "train": (audit.get("split_counts") or {}).get("train"),
                "val": (audit.get("split_counts") or {}).get("val"),
                "test": (audit.get("split_counts") or {}).get("test"),
                "train_imbalance_ratio": audit.get("train_imbalance_ratio"),
                "near_cross_split_pairs": audit.get("near_cross_split_pair_count"),
                "exact_cross_split_groups": audit.get("exact_cross_split_group_count"),
            },
        },
        "model": {
            "name": model.get("name"),
            "width": model.get("width"),
            "dropout": model.get("dropout"),
            "num_classes": model.get("num_classes"),
            "parameter_count": model.get("parameter_count"),
            "image_size": (metadata.get("preprocessing") or {}).get("image_size"),
        },
        "training": {
            "seed": training.get("seed"),
            "epochs_requested": training.get("epochs_requested"),
            "epochs_run": training.get("epochs_run"),
            "batch_size": training.get("batch_size"),
            "learning_rate": training.get("learning_rate"),
            "weight_decay": training.get("weight_decay"),
            "device": training.get("device"),
            "class_strategy": training.get("class_strategy"),
            "loss": training.get("loss") or "cross_entropy",
            "scheduler": training.get("scheduler") or "none",
            "source_note": "Fields missing from bundle metadata are shown with training defaults.",
        },
        "augmentation": augmentation,
        "class_distribution": distribution,
        "failure_analysis": _failure_analysis(metadata, metrics, distribution),
        "reproducibility": _reproducibility(metadata),
        "curves": [_metric_row(row) for row in metrics.get("history", [])],
        "confusion": {
            "labels": [
                {
                    "index": item.get("index"),
                    "class_id": item.get("ip102_id"),
                    "name": item.get("common_name_en") or item.get("dataset_label"),
                }
                for item in classes
            ],
            "matrix": matrix,
            "top_pairs": _confusion_pairs(matrix, classes),
        },
        "best_validation": {
            "epoch": best.get("epoch"),
            "loss": best.get("loss"),
            "samples": best.get("samples"),
            "top1_accuracy": best.get("top1_accuracy"),
            "top3_accuracy": best.get("top3_accuracy"),
            "macro_f1": best.get("macro_f1"),
            "balanced_accuracy": best.get("balanced_accuracy"),
        },
    }


def create_app(
    service: InferenceService | None = None,
    *,
    settings: InferenceSettings | None = None,
    review_store: ReviewStore | None = None,
) -> FastAPI:
    settings = settings or InferenceSettings.from_env()
    app = FastAPI(
        title="PestScope IP102",
        version="0.2.0",
        description="IP102 pest-image triage API backed by a versioned CNN model bundle.",
    )
    app.state.service = service
    app.state.settings = settings
    app.state.review_store = review_store
    app.state.examples = load_demo_examples(
        settings.class_review,
        dataset_root=settings.demo_dataset_root,
        manifest_path=settings.demo_manifest,
    )

    def get_service() -> InferenceService:
        if app.state.service is None:
            bundle_dir = settings.model_bundle
            if not (bundle_dir / "metadata.json").is_file():
                if not settings.allow_demo_model:
                    raise PredictionError(
                        f"Model bundle is missing: {bundle_dir}. "
                        "Train or mount a bundle, or enable PESTSCOPE_ALLOW_DEMO_MODEL."
                    )
                bundle_dir = bundle_dir.with_name("pestnet_s_demo")
                ensure_demo_bundle(
                    bundle_dir=bundle_dir,
                    class_review_path=settings.class_review,
                )
            app.state.service = InferenceService.from_bundle(
                bundle_dir,
                device=settings.device,
                accept_threshold=settings.accept_threshold,
                uncertain_threshold=settings.uncertain_threshold,
            )
        return app.state.service

    def get_review_store() -> ReviewStore:
        if app.state.review_store is None:
            app.state.review_store = ReviewStore(settings.review_db)
        return app.state.review_store

    async def read_upload(file: UploadFile) -> bytes:
        limit = settings.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds {settings.max_upload_mb} MB",
            )
        return content

    def example_by_id(example_id: str) -> DemoExample:
        for example in app.state.examples:
            if example.id == example_id:
                return example
        raise HTTPException(status_code=404, detail="Example not found")

    @app.get("/health")
    def legacy_health() -> dict:
        current = get_service()
        return {
            "status": "ok",
            "ready": current.ready,
            "model_version": current.metadata.get("run_id"),
            "demo_model": bool(current.metadata.get("demo_model", False)),
        }

    @app.get("/api/v1/health/live")
    def live() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/health/ready")
    def ready() -> dict:
        try:
            current = get_service()
        except PredictionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ready",
            "model_version": current.metadata.get("run_id"),
            "demo_model": bool(current.metadata.get("demo_model", False)),
        }

    @app.get("/api/v1/model")
    def model_card() -> dict:
        return get_service().model_card()

    @app.get("/api/v1/experiments/current")
    def current_experiment() -> dict:
        return experiment_evidence_payload(get_service().metadata)

    @app.get("/api/v1/failure-cases/{class_id}/image", response_class=Response)
    def failure_case_image(class_id: int) -> FileResponse:
        for record in _external_benchmark_records():
            if _safe_int(record.get("class_id")) != class_id:
                continue
            image_path = _project_path(record.get("image_path"))
            if image_path and image_path.is_file():
                media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
                return FileResponse(image_path, media_type=media_type)
        raise HTTPException(status_code=404, detail="Failure case image not found")

    @app.get("/api/v1/examples")
    def examples() -> dict:
        return {"examples": [example.to_dict() for example in app.state.examples]}

    @app.get("/api/v1/examples/{example_id}/image", response_class=Response)
    def example_image(example_id: str) -> Response:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=86400",
                "X-Image-License": example.license,
                "X-Image-Provider": example.provider,
            },
        )

    @app.post("/api/v1/examples/{example_id}/predict")
    def predict_example(
        example_id: str,
        top_k: int = Query(default=3, ge=1, le=10),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().predict(image, top_k=top_k)
        result["example"] = example.to_dict()
        return result

    @app.get("/api/v1/examples/{example_id}/stem-activations")
    def stem_activations(
        example_id: str,
        channel_count: int = Query(default=4, ge=1, le=8),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().stem_feature_maps(image, channel_count=channel_count)
        result["example"] = example.to_dict()
        return result

    @app.get("/api/v1/examples/{example_id}/residual32-activations")
    def residual32_activations(
        example_id: str,
        channel_count: int = Query(default=3, ge=1, le=6),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().residual32_feature_maps(image, channel_count=channel_count)
        result["example"] = example.to_dict()
        return result

    @app.get("/api/v1/examples/{example_id}/residual64-activations")
    def residual64_activations(
        example_id: str,
        channel_count: int = Query(default=3, ge=1, le=6),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().residual64_feature_maps(image, channel_count=channel_count)
        result["example"] = example.to_dict()
        return result

    @app.get("/api/v1/examples/{example_id}/residual128-activations")
    def residual128_activations(
        example_id: str,
        channel_count: int = Query(default=3, ge=1, le=6),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().residual128_feature_maps(image, channel_count=channel_count)
        result["example"] = example.to_dict()
        return result

    @app.get("/api/v1/examples/{example_id}/attention-activations")
    def attention_activations(
        example_id: str,
        channel_count: int = Query(default=4, ge=1, le=8),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().attention_feature_maps(image, channel_count=channel_count)
        result["example"] = example.to_dict()
        return result

    @app.get("/api/v1/examples/{example_id}/global-pool-activations")
    def global_pool_activations(
        example_id: str,
        channel_count: int = Query(default=5, ge=1, le=10),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().global_pool_features(image, channel_count=channel_count)
        result["example"] = example.to_dict()
        return result

    @app.get("/api/v1/examples/{example_id}/decision-gate")
    def decision_gate(
        example_id: str,
        top_k: int = Query(default=5, ge=2, le=10),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().decision_gate_features(image, top_k=top_k)
        result["example"] = example.to_dict()
        return result

    @app.post("/api/v1/predictions")
    async def predict_upload(
        file: UploadFile = File(...),
        top_k: int = Query(default=3, ge=1, le=10),
    ) -> dict:
        try:
            image = image_from_upload(
                await read_upload(file),
                max_upload_mb=settings.max_upload_mb,
                max_pixels=settings.max_pixels,
            )
            return get_service().predict(image, top_k=top_k)
        except PredictionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/reviews", status_code=status.HTTP_201_CREATED)
    def create_review(payload: ReviewRequest) -> dict:
        return get_review_store().add(
            prediction_id=payload.prediction_id,
            decision=payload.decision,
            predicted_class_id=payload.predicted_class_id,
            corrected_class_id=payload.corrected_class_id,
            note=payload.note,
            image_consent=payload.image_consent,
        )

    @app.get("/api/v1/reviews/summary")
    def review_summary() -> dict:
        return get_review_store().summary()

    @app.get("/", include_in_schema=False)
    def web_app() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "index.html")

    @app.get("/app.js", include_in_schema=False)
    def web_script() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "app.js", media_type="text/javascript")

    @app.get("/styles.css", include_in_schema=False)
    def web_styles() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "styles.css", media_type="text/css")

    return app


app = create_app()
