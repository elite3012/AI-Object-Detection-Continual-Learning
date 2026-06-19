from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from models.adaptive_service import AdaptiveVisionService
from models.prototype_memory import PrototypeMemory


class MeanColorEmbedder:
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        vectors = []
        for image in images:
            vector = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            vectors.append(vector / np.linalg.norm(vector))
        return np.stack(vectors)


def solid_color(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (16, 16), color)


def test_service_teaches_predicts_and_accepts_feedback() -> None:
    service = AdaptiveVisionService(
        embedder=MeanColorEmbedder(),
        memory=PrototypeMemory(),
        confidence_threshold=0.8,
    )
    service.teach("red", [solid_color((255, 0, 0))])
    service.teach("blue", [solid_color((0, 0, 255))])

    known = service.predict(solid_color((240, 10, 0)))
    unknown = service.predict(solid_color((128, 128, 128)))
    feedback = service.feedback("gray", solid_color((128, 128, 128)))

    assert known.label == "red"
    assert not known.is_unknown
    assert unknown.label is None
    assert unknown.is_unknown
    assert feedback["total_examples"] == 1
    assert service.metrics()["observations"] == 2
