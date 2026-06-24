from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    state_path: Path = Path("artifacts/prototypes.json")
    confidence_threshold: float = 0.90
    drift_window_size: int = 500
    max_upload_mb: int = 10

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_name=os.getenv("VISION_MODEL_NAME", cls.model_name),
            device=os.getenv("VISION_DEVICE", cls.device),
            state_path=Path(os.getenv("VISION_STATE_PATH", str(cls.state_path))),
            confidence_threshold=float(
                os.getenv("VISION_CONFIDENCE_THRESHOLD", cls.confidence_threshold)
            ),
            drift_window_size=int(os.getenv("VISION_DRIFT_WINDOW_SIZE", cls.drift_window_size)),
            max_upload_mb=int(os.getenv("VISION_MAX_UPLOAD_MB", cls.max_upload_mb)),
        )
