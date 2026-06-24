from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceSettings:
    model_bundle: Path = Path("artifacts/models/pestnet_s_latest")
    class_review: Path = Path("configs/data/ip102_class_review.yaml")
    device: str = "auto"
    max_upload_mb: int = 10
    max_pixels: int = 16_000_000
    accept_threshold: float = 0.55
    uncertain_threshold: float = 0.25
    allow_demo_model: bool = True
    fetch_demo_images: bool = True
    demo_cache_dir: Path = Path("artifacts/demo_assets")
    review_db: Path = Path("artifacts/reviews.sqlite3")

    @classmethod
    def from_env(cls) -> InferenceSettings:
        return cls(
            model_bundle=Path(os.getenv("PESTSCOPE_MODEL_BUNDLE", str(cls.model_bundle))),
            class_review=Path(os.getenv("PESTSCOPE_CLASS_REVIEW", str(cls.class_review))),
            device=os.getenv("PESTSCOPE_DEVICE", cls.device),
            max_upload_mb=int(os.getenv("PESTSCOPE_MAX_UPLOAD_MB", cls.max_upload_mb)),
            max_pixels=int(os.getenv("PESTSCOPE_MAX_PIXELS", cls.max_pixels)),
            accept_threshold=float(os.getenv("PESTSCOPE_ACCEPT_THRESHOLD", cls.accept_threshold)),
            uncertain_threshold=float(
                os.getenv("PESTSCOPE_UNCERTAIN_THRESHOLD", cls.uncertain_threshold)
            ),
            allow_demo_model=os.getenv("PESTSCOPE_ALLOW_DEMO_MODEL", "true").lower()
            in {"1", "true", "yes"},
            fetch_demo_images=os.getenv("PESTSCOPE_FETCH_DEMO_IMAGES", "true").lower()
            in {"1", "true", "yes"},
            demo_cache_dir=Path(os.getenv("PESTSCOPE_DEMO_CACHE_DIR", str(cls.demo_cache_dir))),
            review_db=Path(os.getenv("PESTSCOPE_REVIEW_DB", str(cls.review_db))),
        )
