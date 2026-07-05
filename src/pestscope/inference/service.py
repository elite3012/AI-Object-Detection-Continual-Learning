from __future__ import annotations

import json
import time
import uuid
import warnings
from base64 import b64encode
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
        accept_threshold: float | None,
        uncertain_threshold: float | None,
    ) -> InferenceService:
        resolved_device = _resolve_device(device)
        model, metadata = load_model_bundle(bundle_dir, device=str(resolved_device))
        metrics_file = metadata.get("artifact", {}).get("metrics_file")
        if metrics_file:
            metrics_path = bundle_dir / str(metrics_file)
            if metrics_path.is_file():
                metadata["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_hash = metadata.get("artifact", {}).get("model_sha256")
        model_file = metadata.get("artifact", {}).get("model_file", "model.pt")
        if expected_hash and sha256_file(bundle_dir / model_file) != expected_hash:
            raise PredictionError(f"Model bundle hash mismatch: {bundle_dir}")
        thresholds = metadata.get("thresholds", {})
        resolved_accept = (
            accept_threshold
            if accept_threshold is not None
            else float(thresholds.get("accepted", 0.55))
        )
        resolved_uncertain = (
            uncertain_threshold
            if uncertain_threshold is not None
            else float(thresholds.get("uncertain", 0.25))
        )
        if not 0 <= resolved_uncertain <= resolved_accept <= 1:
            raise PredictionError("Invalid inference thresholds")
        return cls(
            model=model,
            metadata=metadata,
            device=resolved_device,
            accept_threshold=resolved_accept,
            uncertain_threshold=resolved_uncertain,
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
            "calibration": self.metadata.get("calibration"),
            "metrics": self.metadata.get("metrics", {}),
            "training": self.metadata.get("training", {}),
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

    def stem_feature_maps(self, image: Image.Image, *, channel_count: int = 4) -> dict:
        if not hasattr(self.model, "features") or len(self.model.features) == 0:
            raise PredictionError("The active model does not expose a stem feature block")
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.features[0](tensor)
        maps = features[0].detach().cpu()
        flattened = maps.flatten(1)
        scores = flattened.std(dim=1)
        selected = torch.topk(scores, min(channel_count, maps.shape[0])).indices.tolist()
        return {
            "layer": "features.0",
            "operation": "Conv2d(3->32, 3x3, stride=2) + BatchNorm + SiLU",
            "input_shape": [3, int(tensor.shape[-2]), int(tensor.shape[-1])],
            "output_shape": [int(value) for value in maps.shape],
            "channels": [
                {
                    "index": int(index),
                    "energy": round(float(scores[index]), 4),
                    "image": _activation_png(maps[index]),
                }
                for index in selected
            ],
        }

    def residual32_feature_maps(self, image: Image.Image, *, channel_count: int = 3) -> dict:
        if not hasattr(self.model, "features") or len(self.model.features) < 2:
            raise PredictionError("The active model does not expose a residual-32 block")
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            stem_features = self.model.features[0](tensor)
            residual_features = self.model.features[1](stem_features)
        before = stem_features[0].detach().cpu()
        after = residual_features[0].detach().cpu()
        if before.shape == after.shape:
            change = after - before
            scores = change.flatten(1).std(dim=1)
        else:
            scores = after.flatten(1).std(dim=1)
        selected = torch.topk(scores, min(channel_count, after.shape[0])).indices.tolist()
        return {
            "layer": "features.1",
            "operation": (
                "ResidualBlock(32->32, stride=1): Conv-BN-SiLU -> Conv-BN "
                "+ identity shortcut -> SiLU"
            ),
            "shortcut": "identity",
            "input_shape": [int(value) for value in before.shape],
            "output_shape": [int(value) for value in after.shape],
            "channels": [
                {
                    "index": int(index),
                    "change": round(float(scores[index]), 4),
                    "before_image": _activation_png(before[index]),
                    "after_image": _activation_png(after[index]),
                }
                for index in selected
            ],
        }

    def residual64_feature_maps(self, image: Image.Image, *, channel_count: int = 3) -> dict:
        if not hasattr(self.model, "features") or len(self.model.features) < 3:
            raise PredictionError("The active model does not expose a residual-64 block")
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        block = self.model.features[2]
        if not all(hasattr(block, name) for name in ("body", "shortcut", "activation")):
            raise PredictionError("The residual-64 block does not expose branch internals")
        with torch.no_grad():
            stem_features = self.model.features[0](tensor)
            residual32_features = self.model.features[1](stem_features)
            branch = block.attention(block.body(residual32_features))
            shortcut = block.shortcut(residual32_features)
            output = block.activation(branch + shortcut)
        branch_maps = branch[0].detach().cpu()
        shortcut_maps = shortcut[0].detach().cpu()
        output_maps = output[0].detach().cpu()
        scores = output_maps.flatten(1).std(dim=1)
        selected = torch.topk(scores, min(channel_count, output_maps.shape[0])).indices.tolist()
        return {
            "layer": "features.2",
            "operation": "ResidualBlock(32->64, stride=2): residual branch + projection shortcut",
            "shortcut": "projection 1x1 stride=2",
            "input_shape": [int(value) for value in residual32_features.shape[1:]],
            "branch_shape": [int(value) for value in branch_maps.shape],
            "shortcut_shape": [int(value) for value in shortcut_maps.shape],
            "output_shape": [int(value) for value in output_maps.shape],
            "channels": [
                {
                    "index": int(index),
                    "energy": round(float(scores[index]), 4),
                    "branch_image": _activation_png(branch_maps[index]),
                    "shortcut_image": _activation_png(shortcut_maps[index]),
                    "output_image": _activation_png(output_maps[index]),
                }
                for index in selected
            ],
        }

    def residual128_feature_maps(self, image: Image.Image, *, channel_count: int = 3) -> dict:
        if not hasattr(self.model, "features") or len(self.model.features) < 5:
            raise PredictionError("The active model does not expose a residual-128 block")
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        block = self.model.features[4]
        if not all(hasattr(block, name) for name in ("body", "shortcut", "activation")):
            raise PredictionError("The residual-128 block does not expose branch internals")
        with torch.no_grad():
            features = tensor
            for layer in self.model.features[:4]:
                features = layer(features)
            branch = block.attention(block.body(features))
            shortcut = block.shortcut(features)
            output = block.activation(branch + shortcut)
        branch_maps = branch[0].detach().cpu()
        shortcut_maps = shortcut[0].detach().cpu()
        output_maps = output[0].detach().cpu()
        scores = output_maps.flatten(1).std(dim=1)
        selected = torch.topk(scores, min(channel_count, output_maps.shape[0])).indices.tolist()
        return {
            "layer": "features.4",
            "operation": (
                "ResidualBlock(64->128, stride=2): deeper shape features + projection shortcut"
            ),
            "shortcut": "projection 1x1 stride=2",
            "input_shape": [int(value) for value in features.shape[1:]],
            "branch_shape": [int(value) for value in branch_maps.shape],
            "shortcut_shape": [int(value) for value in shortcut_maps.shape],
            "output_shape": [int(value) for value in output_maps.shape],
            "channels": [
                {
                    "index": int(index),
                    "energy": round(float(scores[index]), 4),
                    "branch_image": _activation_png(branch_maps[index]),
                    "shortcut_image": _activation_png(shortcut_maps[index]),
                    "output_image": _activation_png(output_maps[index]),
                }
                for index in selected
            ],
        }

    def attention_feature_maps(self, image: Image.Image, *, channel_count: int = 4) -> dict:
        if not hasattr(self.model, "features") or len(self.model.features) < 7:
            raise PredictionError("The active model does not expose an attention block")
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        block = self.model.features[6]
        if not all(
            hasattr(block, name) for name in ("body", "attention", "shortcut", "activation")
        ):
            raise PredictionError("The attention block does not expose branch internals")
        if not all(hasattr(block.attention, name) for name in ("pool", "gate")):
            raise PredictionError("The active attention block is not squeeze-excitation")
        with torch.no_grad():
            features = tensor
            for layer in self.model.features[:6]:
                features = layer(features)
            branch = block.body(features)
            gates = block.attention.gate(block.attention.pool(branch))
            attended = branch * gates
            shortcut = block.shortcut(features)
            output = block.activation(attended + shortcut)
        branch_maps = branch[0].detach().cpu()
        attended_maps = attended[0].detach().cpu()
        gate_values = gates[0, :, 0, 0].detach().cpu()
        output_maps = output[0].detach().cpu()
        selected = torch.topk(gate_values, min(channel_count, gate_values.numel())).indices.tolist()
        return {
            "layer": "features.6.attention",
            "operation": "SqueezeExcitation: global average pool -> channel MLP -> sigmoid gates",
            "input_shape": [int(value) for value in features.shape[1:]],
            "branch_shape": [int(value) for value in branch_maps.shape],
            "gate_shape": [int(gate_values.numel()), 1, 1],
            "output_shape": [int(value) for value in output_maps.shape],
            "gate_summary": {
                "min": round(float(gate_values.min()), 4),
                "mean": round(float(gate_values.mean()), 4),
                "max": round(float(gate_values.max()), 4),
            },
            "channels": [
                {
                    "index": int(index),
                    "gate": round(float(gate_values[index]), 4),
                    "before_image": _activation_png(branch_maps[index]),
                    "after_image": _activation_png(attended_maps[index]),
                }
                for index in selected
            ],
        }

    def global_pool_features(self, image: Image.Image, *, channel_count: int = 5) -> dict:
        if not hasattr(self.model, "features") or not hasattr(self.model, "head"):
            raise PredictionError("The active model does not expose a feature extractor and head")
        if len(self.model.head) == 0:
            raise PredictionError("The active model head does not expose a pooling layer")
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature_tensor = self.model.features(tensor)
            pooled = self.model.head[0](feature_tensor)
        maps = feature_tensor[0].detach().cpu()
        vector = pooled[0, :, 0, 0].detach().cpu()
        scores = vector.abs()
        selected = torch.topk(scores, min(channel_count, vector.numel())).indices.tolist()
        return {
            "layer": "head.0",
            "operation": "AdaptiveAvgPool2d(1): average each channel across all spatial cells",
            "input_shape": [int(value) for value in maps.shape],
            "output_shape": [int(vector.numel())],
            "pooling": "mean over H x W",
            "channels": [
                {
                    "index": int(index),
                    "pooled_value": round(float(vector[index]), 4),
                    "mean": round(float(maps[index].mean()), 4),
                    "max": round(float(maps[index].max()), 4),
                    "image": _activation_png(maps[index]),
                }
                for index in selected
            ],
        }

    def decision_gate_features(self, image: Image.Image, *, top_k: int = 5) -> dict:
        if not hasattr(self.model, "features") or not hasattr(self.model, "head"):
            raise PredictionError("The active model does not expose a feature extractor and head")
        if len(self.model.head) < 4:
            raise PredictionError("The active model head does not expose the classifier path")
        tensor = self._transform()(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature_tensor = self.model.features(tensor)
            pooled = self.model.head[0](feature_tensor)
            vector = self.model.head[1](pooled)
            classifier_input = self.model.head[2](vector)
            logits = self.model.head[3](classifier_input)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
        logits_cpu = logits[0].detach().cpu()
        top_count = min(top_k, probabilities.numel())
        scores, indexes = torch.topk(probabilities, top_count)
        alternatives = []
        for score, index in zip(scores.tolist(), indexes.tolist(), strict=True):
            row = self._alternative(int(index), float(score))
            row["logit"] = round(float(logits_cpu[int(index)]), 4)
            alternatives.append(row)
        confidence = alternatives[0]["confidence"]
        runner_up = alternatives[1]["confidence"] if len(alternatives) > 1 else 0.0
        decision, reason = self._decision(confidence)
        return {
            "layer": "head.3 + softmax",
            "operation": "Linear classifier -> softmax -> confidence thresholds",
            "vector_shape": [int(vector.shape[-1])],
            "logits_shape": [int(logits.shape[-1])],
            "thresholds": {
                "accepted": self.accept_threshold,
                "uncertain": self.uncertain_threshold,
            },
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "margin": round(float(confidence - runner_up), 4),
            "top_k": alternatives,
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


def _activation_png(feature: torch.Tensor) -> str:
    values = feature.float()
    values = values - values.min()
    maximum = values.max()
    if float(maximum) > 0:
        values = values / maximum
    grayscale = Image.fromarray(values.mul(255).clamp(0, 255).byte().numpy())
    heatmap = ImageOps.colorize(grayscale, black="#102b22", mid="#4d9166", white="#f0ba42")
    output = BytesIO()
    heatmap.save(output, format="PNG", optimize=True)
    encoded = b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
