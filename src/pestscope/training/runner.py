from __future__ import annotations

import csv
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pestscope.modeling import build_model, count_parameters

from .bundle import sha256_file, write_model_bundle
from .config import TrainingConfig
from .dataset import ManifestImageDataset, class_counts, class_index, selected_records
from .metadata import load_class_review
from .metrics import classification_summary
from .transforms import DEFAULT_MEAN, DEFAULT_STD, ImageTransform


@dataclass(frozen=True)
class TrainingOverrides:
    max_epochs: int | None = None
    limit_train_per_class: int | None = None
    limit_val_per_class: int | None = None
    device: str | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    bundle_dir: Path | None = None
    log_progress: bool = False


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _git_state() -> dict:
    def run_git(args: list[str]) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    dirty = bool(run_git(["status", "--porcelain"]))
    return {"commit": commit, "dirty": dirty}


def _class_weights(counts: list[int], device: torch.device) -> torch.Tensor:
    if any(count <= 0 for count in counts):
        raise ValueError(f"Every selected class needs at least one training record: {counts}")
    total = sum(counts)
    weights = [total / (len(counts) * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(
        self,
        *,
        gamma: float,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probabilities = F.softmax(logits, dim=1)
        target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_probability).clamp(min=0.0).pow(self.gamma)
        return (focal_weight * cross_entropy).mean()


def _criterion(
    *,
    loss: str,
    weight: torch.Tensor | None,
    label_smoothing: float,
    focal_gamma: float,
) -> nn.Module:
    if loss == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            weight=weight,
            label_smoothing=label_smoothing,
        )
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def _epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        logits_batches.append(logits.detach().cpu())
        target_batches.append(targets.detach().cpu())

    sample_count = sum(batch.numel() for batch in target_batches)
    summary = classification_summary(
        torch.cat(logits_batches),
        torch.cat(target_batches),
        num_classes=num_classes,
    )
    return total_loss / sample_count, summary


def _write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "train_loss",
        "train_macro_f1",
        "train_top1_accuracy",
        "val_loss",
        "val_macro_f1",
        "val_top1_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_epoch_checkpoint(
    *,
    run_dir: Path,
    model: nn.Module,
    history: list[dict],
    metrics: dict,
) -> None:
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    _write_history(run_dir / "history.csv", history)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_training(
    config: TrainingConfig,
    *,
    overrides: TrainingOverrides | None = None,
) -> dict:
    overrides = overrides or TrainingOverrides()
    _set_seed(config.training.seed)
    device = _resolve_device(overrides.device or config.training.device)
    selected_ids = config.data.selected_class_ids
    id_to_index = class_index(selected_ids)
    reviewed_classes = load_class_review(config.data.class_review_path)
    missing_review = sorted(set(selected_ids) - set(reviewed_classes))
    if missing_review:
        raise ValueError(f"Selected classes are missing from review config: {missing_review}")

    train_records = selected_records(
        config.data.manifest_path,
        split="train",
        selected_class_ids=selected_ids,
        limit_per_class=overrides.limit_train_per_class,
    )
    val_records = selected_records(
        config.data.manifest_path,
        split="val",
        selected_class_ids=selected_ids,
        limit_per_class=overrides.limit_val_per_class,
    )
    if not train_records or not val_records:
        raise ValueError("Training requires non-empty train and validation records")

    train_dataset = ManifestImageDataset(
        train_records,
        dataset_root=config.data.dataset_root,
        class_to_index=id_to_index,
        transform=ImageTransform(
            config.data.image_size,
            train=True,
            crop_scale=config.augmentation.crop_scale,
            hflip_probability=config.augmentation.hflip_probability,
            rotation_degrees=config.augmentation.rotation_degrees,
            color_jitter=config.augmentation.color_jitter,
        ),
    )
    val_dataset = ManifestImageDataset(
        val_records,
        dataset_root=config.data.dataset_root,
        class_to_index=id_to_index,
        transform=ImageTransform(config.data.image_size, train=False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=overrides.batch_size or config.training.batch_size,
        shuffle=True,
        num_workers=(
            overrides.num_workers
            if overrides.num_workers is not None
            else config.training.num_workers
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=overrides.batch_size or config.training.batch_size,
        shuffle=False,
        num_workers=(
            overrides.num_workers
            if overrides.num_workers is not None
            else config.training.num_workers
        ),
    )

    model = build_model(
        config.model.name,
        num_classes=len(selected_ids),
        width=config.model.width,
        dropout=config.model.dropout,
    ).to(device)
    weight = (
        _class_weights(class_counts(train_records, selected_ids), device)
        if config.training.class_strategy == "weighted_loss"
        else None
    )
    criterion = _criterion(
        loss=config.training.loss,
        weight=weight,
        label_smoothing=config.training.label_smoothing,
        focal_gamma=config.training.focal_gamma,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.outputs.run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = {"macro_f1": -1.0}
    history = []
    epoch_count = overrides.max_epochs or config.training.epochs
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epoch_count,
            eta_min=config.training.min_learning_rate,
        )
        if config.training.scheduler == "cosine"
        else None
    )
    for epoch in range(1, epoch_count + 1):
        train_loss, train_metrics = _epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=len(selected_ids),
        )
        val_loss, val_metrics = _epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            num_classes=len(selected_ids),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_metrics["macro_f1"],
            "train_top1_accuracy": train_metrics["top1_accuracy"],
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
        }
        history.append(row)
        if val_metrics["macro_f1"] >= best_val["macro_f1"]:
            best_val = {"loss": val_loss, **val_metrics, "epoch": epoch}
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            _write_epoch_checkpoint(
                run_dir=run_dir,
                model=model,
                history=history,
                metrics={
                    "schema_version": 1,
                    "run_id": run_id,
                    "best_validation": best_val,
                    "history": history,
                    "checkpoint_status": "in_progress",
                },
            )
        if scheduler is not None:
            scheduler.step()
        if overrides.log_progress:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "epochs": epoch_count,
                        "train_loss": round(train_loss, 6),
                        "train_macro_f1": round(train_metrics["macro_f1"], 6),
                        "val_loss": round(val_loss, 6),
                        "val_macro_f1": round(val_metrics["macro_f1"], 6),
                        "val_top1_accuracy": round(val_metrics["top1_accuracy"], 6),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()

    class_rows = [
        {"index": id_to_index[class_id], **reviewed_classes[class_id].to_dict()}
        for class_id in selected_ids
    ]
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "best_validation": best_val,
        "history": history,
    }
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "git": _git_state(),
        "dataset": {
            "root": str(config.data.dataset_root),
            "manifest": str(config.data.manifest_path),
            "manifest_sha256": sha256_file(config.data.manifest_path),
            "selected_class_ids": list(selected_ids),
            "train_records": len(train_records),
            "val_records": len(val_records),
            "train_limit_per_class": overrides.limit_train_per_class,
            "val_limit_per_class": overrides.limit_val_per_class,
        },
        "model": {
            "name": config.model.name,
            "width": config.model.width,
            "dropout": config.model.dropout,
            "num_classes": len(selected_ids),
            "parameter_count": count_parameters(model),
        },
        "preprocessing": {
            "image_size": config.data.image_size,
            "mean": list(DEFAULT_MEAN),
            "std": list(DEFAULT_STD),
        },
        "classes": class_rows,
        "training": {
            "seed": config.training.seed,
            "epochs_requested": config.training.epochs,
            "epochs_run": epoch_count,
            "batch_size": config.training.batch_size,
            "batch_size_effective": overrides.batch_size or config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "min_learning_rate": config.training.min_learning_rate,
            "weight_decay": config.training.weight_decay,
            "device": str(device),
            "num_workers_effective": (
                overrides.num_workers
                if overrides.num_workers is not None
                else config.training.num_workers
            ),
            "class_strategy": config.training.class_strategy,
            "loss": config.training.loss,
            "label_smoothing": config.training.label_smoothing,
            "focal_gamma": config.training.focal_gamma,
            "scheduler": config.training.scheduler,
        },
        "augmentation": {
            "crop_scale": list(config.augmentation.crop_scale),
            "hflip_probability": config.augmentation.hflip_probability,
            "rotation_degrees": config.augmentation.rotation_degrees,
            "color_jitter": config.augmentation.color_jitter,
        },
    }

    bundle_dir = overrides.bundle_dir or config.outputs.bundle_dir
    bundle = write_model_bundle(
        bundle_dir=bundle_dir,
        model=model,
        metadata=metadata,
        metrics=metrics,
    )
    _write_history(run_dir / "history.csv", history)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "bundle": bundle,
        "best_validation": best_val,
        "parameter_count": metadata["model"]["parameter_count"],
    }
