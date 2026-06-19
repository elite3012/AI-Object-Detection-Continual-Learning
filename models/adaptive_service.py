from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from .drift import DriftMonitor
from .embeddings import ImageEmbedder
from .prototype_memory import Match, PrototypeMemory

LABEL_PATTERN = re.compile(r"^[\w][\w .-]{0,79}$", flags=re.UNICODE)


class NoClassesError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str | None
    is_unknown: bool
    threshold: float
    matches: list[Match]


class AdaptiveVisionService:
    def __init__(
        self,
        embedder: ImageEmbedder,
        memory: PrototypeMemory,
        monitor: DriftMonitor | None = None,
        confidence_threshold: float = 0.55,
    ) -> None:
        if not -1.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between -1 and 1")
        self.embedder = embedder
        self.memory = memory
        self.monitor = monitor or DriftMonitor()
        self.confidence_threshold = confidence_threshold

    def teach(self, label: str, images: Sequence[Image.Image]) -> dict:
        clean_label = self._validate_label(label)
        if not images:
            raise ValueError("At least one example image is required")

        embeddings = self.embedder.embed(images)
        prototype = self.memory.upsert(clean_label, embeddings)
        self.memory.save()
        return {
            "label": prototype.label,
            "examples_added": len(images),
            "total_examples": prototype.count,
            "updated_at": prototype.updated_at,
        }

    def predict(self, image: Image.Image, top_k: int = 3) -> Prediction:
        if not self.memory.classes():
            raise NoClassesError("Teach at least one class before requesting predictions")

        embedding = self.embedder.embed([image])[0]
        matches = self.memory.match(embedding, top_k=top_k)
        top_match = matches[0]
        is_unknown = top_match.similarity < self.confidence_threshold
        self.monitor.record(top_match.similarity, is_unknown)
        return Prediction(
            label=None if is_unknown else top_match.label,
            is_unknown=is_unknown,
            threshold=self.confidence_threshold,
            matches=matches,
        )

    def feedback(self, label: str, image: Image.Image) -> dict:
        return self.teach(label, [image])

    def delete_class(self, label: str) -> bool:
        clean_label = self._validate_label(label)
        deleted = self.memory.delete(clean_label)
        if deleted:
            self.memory.save()
        return deleted

    def classes(self) -> list[dict]:
        return self.memory.classes()

    def metrics(self) -> dict:
        classes = self.memory.classes()
        return {
            "class_count": len(classes),
            "example_count": sum(item["examples"] for item in classes),
            "confidence_threshold": self.confidence_threshold,
            **self.monitor.summary(),
        }

    @staticmethod
    def _validate_label(label: str) -> str:
        clean_label = label.strip()
        if not LABEL_PATTERN.fullmatch(clean_label):
            raise ValueError(
                "Label must be 1-80 characters and contain only letters, numbers, spaces, dot, "
                "underscore, or hyphen"
            )
        return clean_label
