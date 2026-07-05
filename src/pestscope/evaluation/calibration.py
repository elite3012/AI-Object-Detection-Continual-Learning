from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pestscope.data.manifest import ManifestRecord, read_manifest
from pestscope.training.bundle import load_model_bundle, sha256_file, write_bundle_thresholds
from pestscope.training.dataset import ManifestImageDataset, class_index
from pestscope.training.metrics import classification_summary
from pestscope.training.transforms import ImageTransform


@dataclass(frozen=True)
class CalibrationPolicy:
    target_accept_precision: float = 0.70
    min_accept_coverage: float = 0.05
    ood_reject_quantile: float = 0.95


def _limited(records: list[ManifestRecord], limit_per_class: int | None) -> list[ManifestRecord]:
    if limit_per_class is None:
        return records
    if limit_per_class < 1:
        raise ValueError("limit_per_class must be positive")
    counts: Counter[int] = Counter()
    limited = []
    for record in records:
        if counts[record.class_id] >= limit_per_class:
            continue
        counts[record.class_id] += 1
        limited.append(record)
    return limited


def _split_records(
    manifest_path: Path,
    *,
    split: str,
    selected_class_ids: tuple[int, ...],
    id_limit_per_class: int | None,
    ood_limit_per_class: int | None,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    selected = set(selected_class_ids)
    all_records = [
        record
        for record in read_manifest(manifest_path)
        if record.split == split and record.status == "ok"
    ]
    id_records = [record for record in all_records if record.class_id in selected]
    ood_records = [record for record in all_records if record.class_id not in selected]
    return _limited(id_records, id_limit_per_class), _limited(ood_records, ood_limit_per_class)


def _score_records(
    *,
    model: torch.nn.Module,
    records: list[ManifestRecord],
    dataset_root: Path,
    selected_class_ids: tuple[int, ...],
    transform: ImageTransform,
    batch_size: int,
    device: torch.device,
    include_targets: bool,
) -> tuple[list[dict], torch.Tensor | None, torch.Tensor | None]:
    if not records:
        return [], None, None

    id_to_index = class_index(selected_class_ids)
    dataset = ManifestImageDataset(
        records,
        dataset_root=dataset_root,
        class_to_index={record.class_id: id_to_index.get(record.class_id, 0) for record in records},
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scored = []
    logits_batches = []
    target_batches = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            top_scores, top_indexes = torch.max(probabilities, dim=1)
            batch_records = records[offset : offset + images.size(0)]
            offset += images.size(0)
            if include_targets:
                logits_batches.append(logits.detach().cpu())
                target_batches.append(targets.detach().cpu())
            for row_index, record in enumerate(batch_records):
                predicted_index = int(top_indexes[row_index])
                predicted_class_id = selected_class_ids[predicted_index]
                target_index = id_to_index.get(record.class_id)
                scored.append(
                    {
                        "image_id": record.image_id,
                        "path": record.path,
                        "class_id": record.class_id,
                        "target_index": target_index,
                        "predicted_index": predicted_index,
                        "predicted_class_id": predicted_class_id,
                        "confidence": float(top_scores[row_index]),
                        "correct": bool(target_index == predicted_index)
                        if include_targets
                        else None,
                    }
                )
    if not include_targets:
        return scored, None, None
    return scored, torch.cat(logits_batches), torch.cat(target_batches)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _acceptance_rows(id_scores: list[dict]) -> list[dict]:
    candidates = sorted({round(row["confidence"], 4) for row in id_scores})
    rows = []
    for threshold in candidates:
        accepted = [row for row in id_scores if row["confidence"] >= threshold]
        if not accepted:
            continue
        correct = sum(1 for row in accepted if row["correct"])
        rows.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "coverage": len(accepted) / len(id_scores),
                "accepted_precision": correct / len(accepted),
            }
        )
    return rows


def select_thresholds(
    id_scores: list[dict],
    ood_scores: list[dict],
    *,
    policy: CalibrationPolicy | None = None,
) -> dict:
    if not id_scores:
        raise ValueError("At least one in-distribution score is required")
    policy = policy or CalibrationPolicy()
    candidates = _acceptance_rows(id_scores)
    viable = [
        row
        for row in candidates
        if row["accepted_precision"] >= policy.target_accept_precision
        and row["coverage"] >= policy.min_accept_coverage
    ]
    if viable:
        chosen = max(viable, key=lambda row: (row["coverage"], -row["threshold"]))
        met_target = True
    else:
        max_confidence = max(row["confidence"] for row in id_scores)
        chosen = {
            "threshold": min(1.0, round(max_confidence + 0.0001, 4)),
            "accepted": 0,
            "coverage": 0.0,
            "accepted_precision": 0.0,
        }
        met_target = False

    ood_confidences = [row["confidence"] for row in ood_scores]
    ood_quantile = _quantile(ood_confidences, policy.ood_reject_quantile)
    if ood_confidences:
        uncertain = min(chosen["threshold"] * 0.95, ood_quantile)
    else:
        uncertain = chosen["threshold"] * 0.5
    uncertain = max(0.01, min(chosen["threshold"], uncertain))
    return {
        "accepted": round(float(chosen["threshold"]), 4),
        "uncertain": round(float(uncertain), 4),
        "selection": {
            "target_accept_precision": policy.target_accept_precision,
            "min_accept_coverage": policy.min_accept_coverage,
            "ood_reject_quantile": policy.ood_reject_quantile,
            "accepted_precision": chosen["accepted_precision"],
            "accepted_coverage": chosen["coverage"],
            "ood_confidence_quantile": ood_quantile,
            "candidate_count": len(candidates),
            "met_target": met_target,
            "conservative_no_acceptance": not met_target,
        },
    }


def _decision_summary(
    id_scores: list[dict],
    ood_scores: list[dict],
    *,
    accepted: float,
    uncertain: float,
) -> dict:
    accepted_id = [row for row in id_scores if row["confidence"] >= accepted]
    accepted_correct = sum(1 for row in accepted_id if row["correct"])
    unsupported_id = [row for row in id_scores if row["confidence"] < uncertain]
    accepted_ood = [row for row in ood_scores if row["confidence"] >= accepted]
    unsupported_ood = [row for row in ood_scores if row["confidence"] < uncertain]
    return {
        "id_accepted_count": len(accepted_id),
        "id_accepted_precision": accepted_correct / len(accepted_id) if accepted_id else 0.0,
        "id_coverage": len(accepted_id) / len(id_scores) if id_scores else 0.0,
        "id_unsupported_rate": len(unsupported_id) / len(id_scores) if id_scores else 0.0,
        "near_ood_samples": len(ood_scores),
        "near_ood_accepted_rate": len(accepted_ood) / len(ood_scores) if ood_scores else 0.0,
        "near_ood_unsupported_rate": len(unsupported_ood) / len(ood_scores) if ood_scores else 0.0,
    }


def _per_class_summary(id_scores: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in id_scores:
        grouped[row["class_id"]].append(row)
    summary = []
    for class_id, rows in sorted(grouped.items()):
        correct = sum(1 for row in rows if row["correct"])
        summary.append(
            {
                "class_id": class_id,
                "samples": len(rows),
                "top1_accuracy": correct / len(rows),
                "mean_confidence": sum(row["confidence"] for row in rows) / len(rows),
            }
        )
    return summary


def evaluate_bundle(
    *,
    bundle_dir: Path,
    dataset_root: Path,
    manifest_path: Path,
    selected_class_ids: tuple[int, ...],
    split: str = "val",
    batch_size: int = 32,
    device: str = "cpu",
    id_limit_per_class: int | None = None,
    ood_limit_per_class: int | None = None,
    policy: CalibrationPolicy | None = None,
    output_path: Path | None = None,
    write_thresholds: bool = False,
) -> dict:
    if write_thresholds and split == "test":
        raise ValueError("Do not write calibrated thresholds from the official test split")
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    model, metadata = load_model_bundle(bundle_dir, device=str(torch_device))
    preprocessing = metadata["preprocessing"]
    transform = ImageTransform(
        int(preprocessing["image_size"]),
        train=False,
        mean=tuple(float(value) for value in preprocessing["mean"]),
        std=tuple(float(value) for value in preprocessing["std"]),
    )
    id_records, ood_records = _split_records(
        manifest_path,
        split=split,
        selected_class_ids=selected_class_ids,
        id_limit_per_class=id_limit_per_class,
        ood_limit_per_class=ood_limit_per_class,
    )
    id_scores, logits, targets = _score_records(
        model=model,
        records=id_records,
        dataset_root=dataset_root,
        selected_class_ids=selected_class_ids,
        transform=transform,
        batch_size=batch_size,
        device=torch_device,
        include_targets=True,
    )
    ood_scores, _, _ = _score_records(
        model=model,
        records=ood_records,
        dataset_root=dataset_root,
        selected_class_ids=selected_class_ids,
        transform=transform,
        batch_size=batch_size,
        device=torch_device,
        include_targets=False,
    )
    if logits is None or targets is None:
        raise ValueError("No in-distribution validation records were scored")

    thresholds = select_thresholds(id_scores, ood_scores, policy=policy)
    decision = _decision_summary(
        id_scores,
        ood_scores,
        accepted=thresholds["accepted"],
        uncertain=thresholds["uncertain"],
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "model_version": metadata.get("run_id"),
        "split": split,
        "limits": {
            "id_limit_per_class": id_limit_per_class,
            "ood_limit_per_class": ood_limit_per_class,
        },
        "model_sha256": sha256_file(bundle_dir / metadata["artifact"]["model_file"]),
        "classification": classification_summary(
            logits,
            targets,
            num_classes=len(selected_class_ids),
        ),
        "thresholds": thresholds,
        "decision_summary": decision,
        "per_class": _per_class_summary(id_scores),
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_thresholds:
        write_bundle_thresholds(
            bundle_dir,
            accepted=thresholds["accepted"],
            uncertain=thresholds["uncertain"],
            calibration={
                "created_at": result["created_at"],
                "split": split,
                "output_path": str(output_path) if output_path else None,
                "decision_summary": decision,
                "selection": thresholds["selection"],
            },
        )
    return result
