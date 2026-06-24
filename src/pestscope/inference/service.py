from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from pestscope.training.bundle import load_model_bundle, sha256_file
from pestscope.training.transforms import ImageTransform


class PredictionError(ValueError):
    """Raised when an uploaded image cannot be scored."""


@dataclass(frozen=True)
class InferenceService:
    model: torch.nn.Module
    metadata: dict
    device: torch.device
    accept_threshold: float
    uncertain_threshold: float

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: Path,
        *,
        device: str,
        accept_threshold: float,
        uncertain_threshold: float,
    ) -> InferenceService:
        resolved_device = _resolve_device(device)
        model, metadata = load_model_bundle(bundle_dir, device=str(resolved_device))
        expected_hash = metadata.get("artifact", {}).get("model_sha256")
        model_file = metadata.get("artifact", {}).get("model_file", "model.pt")
        if expected_hash and sha256_file(bundle_dir / model_file) != expected_hash:
            raise PredictionError(f"Model bundle hash mismatch: {bundle_dir}")
        return cls(
            model=model,
            metadata=metadata,
            device=resolved_device,
            accept_threshold=accept_threshold,
            uncertain_threshold=uncertain_threshold,
        )

    @property
    def ready(self) -> bool:
        return True

    @property
    def classes(self) -> list[dict]:
        return sorted(self.metadata["classes"], key=lambda item: int(item["index"]))

    def model_card(self) -> dict:
        return {
            "model": self.metadata["model"],
            "dataset": self.metadata.get("dataset", {}),
            "preprocessing": self.metadata["preprocessing"],
            "classes": self.classes,
            "run_id": self.metadata.get("run_id"),
            "created_at": self.metadata.get("created_at"),
            "demo_model": bool(self.metadata.get("demo_model", False)),
            "warning": self.metadata.get("warning"),
            "thresholds": {
                "accepted": self.accept_threshold,
                "uncertain": self.uncertain_threshold,
            },
        }

    def predict(self, image: Image.Image, *, top_k: int = 3) -> dict:
        started = time.perf_counter()
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        top_count = min(top_k, probabilities.numel())
        scores, indexes = torch.topk(probabilities, top_count)
        alternatives = [
            self._alternative(int(index), float(score))
            for score, index in zip(scores.tolist(), indexes.tolist(), strict=True)
        ]
        confidence = alternatives[0]["confidence"]
        decision, reason = self._decision(confidence)
        return {
            "prediction_id": uuid.uuid4().hex,
            "model_version": self.metadata.get("run_id", "unknown"),
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "top_k": alternatives,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "demo_model": bool(self.metadata.get("demo_model", False)),
        }

    def _transform(self) -> ImageTransform:
        preprocessing = self.metadata["preprocessing"]
        return ImageTransform(
            int(preprocessing["image_size"]),
            train=False,
            mean=tuple(float(value) for value in preprocessing["mean"]),
            std=tuple(float(value) for value in preprocessing["std"]),
        )

    def _alternative(self, index: int, confidence: float) -> dict:
        class_row = self.classes[index]
        return {
            "index": index,
            "class_id": int(class_row["ip102_id"]),
            "dataset_label": class_row["dataset_label"],
            "canonical_name": class_row["canonical_name"],
            "common_name_en": class_row["common_name_en"],
            "common_name_vi": class_row["common_name_vi"],
            "stratum": class_row["stratum"],
            "confidence": confidence,
        }

    def _decision(self, confidence: float) -> tuple[str, str]:
        if confidence >= self.accept_threshold:
            return "accepted", "Top class passed the acceptance threshold."
        if confidence >= self.uncertain_threshold:
            return "uncertain", "The model found a possible match but confidence is low."
        return "unsupported", "Image is outside the supported scope or too ambiguous."


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PredictionError("CUDA was requested but is not available")
    return device


def image_from_upload(
    content: bytes,
    *,
    max_upload_mb: int,
    max_pixels: int,
) -> Image.Image:
    if len(content) > max_upload_mb * 1024 * 1024:
        raise PredictionError(f"Image exceeds {max_upload_mb} MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                if width * height > max_pixels:
                    raise PredictionError(f"Image exceeds {max_pixels:,} pixels")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except PredictionError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise PredictionError("Image is too large to process safely") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise PredictionError("File is not a valid image") from exc
