from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image


class ImageEmbedder(Protocol):
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return one L2-normalized embedding per image."""


class CLIPImageEmbedder:
    """Lazy CLIP adapter so the API starts before model weights are loaded."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.requested_device = device
        self._device: str | None = None
        self._model = None
        self._processor = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            import torch
            from transformers import AutoProcessor, CLIPModel

            if self.requested_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.requested_device

            processor = AutoProcessor.from_pretrained(self.model_name)
            model = CLIPModel.from_pretrained(self.model_name)
            model.eval()
            model.to(device)

            self._processor = processor
            self._model = model
            self._device = device

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            raise ValueError("At least one image is required")

        self._ensure_loaded()

        import torch

        rgb_images = [image.convert("RGB") for image in images]
        with self._inference_lock, torch.inference_mode():
            inputs = self._processor(images=rgb_images, return_tensors="pt")
            inputs = {name: value.to(self._device) for name, value in inputs.items()}
            features = self._model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32)
