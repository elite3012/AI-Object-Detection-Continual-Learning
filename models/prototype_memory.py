from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must have a finite, non-zero norm")
    return vector / norm


@dataclass(frozen=True)
class Match:
    label: str
    similarity: float
    examples: int


@dataclass
class ClassPrototype:
    label: str
    count: int
    sum_vector: np.ndarray
    created_at: str
    updated_at: str

    @property
    def centroid(self) -> np.ndarray:
        return _normalize(self.sum_vector)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "count": self.count,
            "sum_vector": self.sum_vector.tolist(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ClassPrototype:
        return cls(
            label=str(data["label"]),
            count=int(data["count"]),
            sum_vector=np.asarray(data["sum_vector"], dtype=np.float32),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
        )


class PrototypeMemory:
    """Thread-safe class memory backed by atomic JSON snapshots."""

    VERSION = 1

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._classes: dict[str, ClassPrototype] = {}
        self._dimension: int | None = None
        self._lock = threading.RLock()

        if self.path and self.path.exists():
            self.load()

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def upsert(self, label: str, embeddings: np.ndarray) -> ClassPrototype:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("Embeddings must have shape [examples, dimensions]")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Embeddings contain non-finite values")

        normalized = np.stack([_normalize(row) for row in matrix])
        now = _utc_now()

        with self._lock:
            if self._dimension is None:
                self._dimension = normalized.shape[1]
            elif normalized.shape[1] != self._dimension:
                raise ValueError(
                    f"Embedding dimension {normalized.shape[1]} does not match "
                    f"memory dimension {self._dimension}"
                )

            current = self._classes.get(label)
            if current is None:
                prototype = ClassPrototype(
                    label=label,
                    count=normalized.shape[0],
                    sum_vector=normalized.sum(axis=0),
                    created_at=now,
                    updated_at=now,
                )
            else:
                current.count += normalized.shape[0]
                current.sum_vector += normalized.sum(axis=0)
                current.updated_at = now
                prototype = current

            self._classes[label] = prototype
            return prototype

    def match(self, embedding: np.ndarray, top_k: int = 3) -> list[Match]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query = _normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        with self._lock:
            if not self._classes:
                return []
            if query.shape[0] != self._dimension:
                raise ValueError(
                    f"Embedding dimension {query.shape[0]} does not match "
                    f"memory dimension {self._dimension}"
                )

            matches = [
                Match(
                    label=prototype.label,
                    similarity=float(np.dot(query, prototype.centroid)),
                    examples=prototype.count,
                )
                for prototype in self._classes.values()
            ]

        return sorted(matches, key=lambda item: item.similarity, reverse=True)[:top_k]

    def delete(self, label: str) -> bool:
        with self._lock:
            deleted = self._classes.pop(label, None) is not None
            if not self._classes:
                self._dimension = None
            return deleted

    def classes(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "label": item.label,
                    "examples": item.count,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                }
                for item in sorted(self._classes.values(), key=lambda value: value.label)
            ]

    def save(self) -> None:
        if self.path is None:
            return

        with self._lock:
            payload = {
                "version": self.VERSION,
                "dimension": self._dimension,
                "classes": [item.to_dict() for item in self._classes.values()],
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)

    def load(self) -> None:
        if self.path is None:
            raise ValueError("Cannot load memory without a path")

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("version") != self.VERSION:
            raise ValueError(f"Unsupported memory version: {payload.get('version')}")

        classes = {
            item.label: item
            for item in (ClassPrototype.from_dict(raw) for raw in payload.get("classes", []))
        }
        dimension = payload.get("dimension")

        for prototype in classes.values():
            if prototype.count < 1:
                raise ValueError(f"Class {prototype.label!r} has an invalid example count")
            if dimension is None or prototype.sum_vector.shape != (int(dimension),):
                raise ValueError(f"Class {prototype.label!r} has an invalid embedding dimension")
            _normalize(prototype.sum_vector)

        with self._lock:
            self._classes = classes
            self._dimension = int(dimension) if dimension is not None else None
