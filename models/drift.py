from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class DriftObservation:
    timestamp: str
    top_similarity: float
    is_unknown: bool


class DriftMonitor:
    """Rolling signal for low-similarity traffic and possible data drift."""

    def __init__(self, window_size: int = 500) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self._observations: deque[DriftObservation] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, top_similarity: float, is_unknown: bool) -> None:
        observation = DriftObservation(
            timestamp=datetime.now(timezone.utc).isoformat(),
            top_similarity=float(top_similarity),
            is_unknown=bool(is_unknown),
        )
        with self._lock:
            self._observations.append(observation)

    def summary(self) -> dict:
        with self._lock:
            observations = list(self._observations)

        if not observations:
            return {
                "window_size": self.window_size,
                "observations": 0,
                "unknown_rate": 0.0,
                "mean_top_similarity": None,
                "p10_top_similarity": None,
                "last_observation_at": None,
            }

        similarities = np.asarray([item.top_similarity for item in observations])
        return {
            "window_size": self.window_size,
            "observations": len(observations),
            "unknown_rate": float(np.mean([item.is_unknown for item in observations])),
            "mean_top_similarity": float(np.mean(similarities)),
            "p10_top_similarity": float(np.percentile(similarities, 10)),
            "last_observation_at": observations[-1].timestamp,
        }
