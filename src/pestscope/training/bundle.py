from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from pestscope.modeling import build_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_bundle(
    *,
    bundle_dir: Path,
    model: nn.Module,
    metadata: dict,
    metrics: dict,
) -> dict:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model_path = bundle_dir / "model.pt"
    metadata_path = bundle_dir / "metadata.json"
    metrics_path = bundle_dir / "metrics.json"

    torch.save(model.state_dict(), model_path)
    model_sha256 = sha256_file(model_path)
    full_metadata = dict(metadata)
    full_metadata["artifact"] = {
        "model_file": model_path.name,
        "model_sha256": model_sha256,
        "metrics_file": metrics_path.name,
    }
    metadata_path.write_text(
        json.dumps(full_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "model": str(model_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "model_sha256": model_sha256,
    }


def load_model_bundle(bundle_dir: Path, *, device: str = "cpu") -> tuple[nn.Module, dict]:
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    model_info = metadata["model"]
    model = build_model(
        model_info["name"],
        num_classes=int(model_info["num_classes"]),
        width=int(model_info["width"]),
        dropout=float(model_info["dropout"]),
    )
    model_path = bundle_dir / metadata["artifact"]["model_file"]
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, metadata


def write_bundle_thresholds(
    bundle_dir: Path,
    *,
    accepted: float,
    uncertain: float,
    calibration: dict,
) -> dict:
    if not 0 <= uncertain <= accepted <= 1:
        raise ValueError("Thresholds must satisfy 0 <= uncertain <= accepted <= 1")
    metadata_path = bundle_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["thresholds"] = {
        "accepted": accepted,
        "uncertain": uncertain,
    }
    metadata["calibration"] = calibration
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
