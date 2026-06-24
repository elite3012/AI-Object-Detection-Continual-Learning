from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class TrainingConfigError(ValueError):
    """Raised when a training configuration cannot be interpreted safely."""


@dataclass(frozen=True)
class TrainingDataConfig:
    dataset_root: Path
    manifest_path: Path
    class_review_path: Path
    selected_class_ids: tuple[int, ...]
    image_size: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    width: int
    dropout: float


@dataclass(frozen=True)
class OptimizerConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    device: str
    num_workers: int
    class_strategy: str


@dataclass(frozen=True)
class OutputConfig:
    run_dir: Path
    bundle_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    data: TrainingDataConfig
    model: ModelConfig
    training: OptimizerConfig
    outputs: OutputConfig


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{name} must be a mapping")
    return value


def _class_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TrainingConfigError("data.selected_class_ids must be a non-empty list")
    class_ids = tuple(int(item) for item in value)
    if len(class_ids) != len(set(class_ids)):
        raise TrainingConfigError("data.selected_class_ids must not contain duplicates")
    if len(class_ids) < 2:
        raise TrainingConfigError("at least two selected classes are required")
    return class_ids


def load_training_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "config")
    data = _mapping(root.get("data"), "data")
    model = _mapping(root.get("model"), "model")
    training = _mapping(root.get("training"), "training")
    outputs = _mapping(root.get("outputs"), "outputs")

    image_size = int(data.get("image_size", 224))
    if image_size < 32:
        raise TrainingConfigError("data.image_size must be at least 32")

    batch_size = int(training.get("batch_size", 32))
    if batch_size < 1:
        raise TrainingConfigError("training.batch_size must be positive")
    epochs = int(training.get("epochs", 1))
    if epochs < 1:
        raise TrainingConfigError("training.epochs must be positive")

    class_strategy = str(training.get("class_strategy", "none"))
    if class_strategy not in {"none", "weighted_loss"}:
        raise TrainingConfigError("training.class_strategy must be none or weighted_loss")

    return TrainingConfig(
        data=TrainingDataConfig(
            dataset_root=Path(str(data.get("dataset_root", "data/raw/ip102/ip102_v1.1"))),
            manifest_path=Path(str(data.get("manifest", "artifacts/data/ip102_manifest.csv"))),
            class_review_path=Path(str(data.get("class_review"))),
            selected_class_ids=_class_ids(data.get("selected_class_ids")),
            image_size=image_size,
        ),
        model=ModelConfig(
            name=str(model.get("name", "pestnet_s")),
            width=int(model.get("width", 32)),
            dropout=float(model.get("dropout", 0.25)),
        ),
        training=OptimizerConfig(
            seed=int(training.get("seed", 2026)),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            device=str(training.get("device", "auto")),
            num_workers=int(training.get("num_workers", 0)),
            class_strategy=class_strategy,
        ),
        outputs=OutputConfig(
            run_dir=Path(str(outputs.get("run_dir", "artifacts/runs/pestnet_s"))),
            bundle_dir=Path(str(outputs.get("bundle_dir", "artifacts/models/pestnet_s_latest"))),
        ),
    )
