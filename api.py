from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.inference.config import InferenceSettings  # noqa: E402
from pestscope.inference.demo_model import ensure_demo_bundle  # noqa: E402
from pestscope.inference.examples import (  # noqa: E402
    DemoExample,
    example_image_bytes,
    load_demo_examples,
)
from pestscope.inference.reviews import ReviewStore  # noqa: E402
from pestscope.inference.service import (  # noqa: E402
    InferenceService,
    PredictionError,
    image_from_upload,
)


class ReviewRequest(BaseModel):
    prediction_id: str = Field(..., min_length=8, max_length=80)
    decision: str = Field(..., min_length=3, max_length=32)
    predicted_class_id: int | None = None
    corrected_class_id: int | None = None
    note: str | None = Field(default=None, max_length=500)
    image_consent: bool = False


def create_app(
    service: InferenceService | None = None,
    *,
    settings: InferenceSettings | None = None,
    review_store: ReviewStore | None = None,
) -> FastAPI:
    settings = settings or InferenceSettings.from_env()
    app = FastAPI(
        title="PestScope IP102",
        version="0.2.0",
        description="IP102 pest-image triage API backed by a versioned CNN model bundle.",
    )
    app.state.service = service
    app.state.settings = settings
    app.state.review_store = review_store
    app.state.examples = load_demo_examples(settings.class_review)

    def get_service() -> InferenceService:
        if app.state.service is None:
            bundle_dir = settings.model_bundle
            if not (bundle_dir / "metadata.json").is_file():
                if not settings.allow_demo_model:
                    raise PredictionError(
                        f"Model bundle is missing: {bundle_dir}. "
                        "Train or mount a bundle, or enable PESTSCOPE_ALLOW_DEMO_MODEL."
                    )
                bundle_dir = bundle_dir.with_name("pestnet_s_demo")
                ensure_demo_bundle(
                    bundle_dir=bundle_dir,
                    class_review_path=settings.class_review,
                )
            app.state.service = InferenceService.from_bundle(
                bundle_dir,
                device=settings.device,
                accept_threshold=settings.accept_threshold,
                uncertain_threshold=settings.uncertain_threshold,
            )
        return app.state.service

    def get_review_store() -> ReviewStore:
        if app.state.review_store is None:
            app.state.review_store = ReviewStore(settings.review_db)
        return app.state.review_store

    async def read_upload(file: UploadFile) -> bytes:
        limit = settings.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds {settings.max_upload_mb} MB",
            )
        return content

    def example_by_id(example_id: str) -> DemoExample:
        for example in app.state.examples:
            if example.id == example_id:
                return example
        raise HTTPException(status_code=404, detail="Example not found")

    @app.get("/health")
    def legacy_health() -> dict:
        current = get_service()
        return {
            "status": "ok",
            "ready": current.ready,
            "model_version": current.metadata.get("run_id"),
            "demo_model": bool(current.metadata.get("demo_model", False)),
        }

    @app.get("/api/v1/health/live")
    def live() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/health/ready")
    def ready() -> dict:
        try:
            current = get_service()
        except PredictionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "status": "ready",
            "model_version": current.metadata.get("run_id"),
            "demo_model": bool(current.metadata.get("demo_model", False)),
        }

    @app.get("/api/v1/model")
    def model_card() -> dict:
        return get_service().model_card()

    @app.get("/api/v1/examples")
    def examples() -> dict:
        return {"examples": [example.to_dict() for example in app.state.examples]}

    @app.get("/api/v1/examples/{example_id}/image", response_class=Response)
    def example_image(example_id: str) -> Response:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=86400",
                "X-Image-License": example.license,
                "X-Image-Provider": example.provider,
            },
        )

    @app.post("/api/v1/examples/{example_id}/predict")
    def predict_example(
        example_id: str,
        top_k: int = Query(default=3, ge=1, le=10),
    ) -> dict:
        example = example_by_id(example_id)
        content = example_image_bytes(
            example,
            cache_dir=settings.demo_cache_dir,
            fetch_external=settings.fetch_demo_images,
        )
        image = image_from_upload(
            content,
            max_upload_mb=settings.max_upload_mb,
            max_pixels=settings.max_pixels,
        )
        result = get_service().predict(image, top_k=top_k)
        result["example"] = example.to_dict()
        return result

    @app.post("/api/v1/predictions")
    async def predict_upload(
        file: UploadFile = File(...),
        top_k: int = Query(default=3, ge=1, le=10),
    ) -> dict:
        try:
            image = image_from_upload(
                await read_upload(file),
                max_upload_mb=settings.max_upload_mb,
                max_pixels=settings.max_pixels,
            )
            return get_service().predict(image, top_k=top_k)
        except PredictionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/reviews", status_code=status.HTTP_201_CREATED)
    def create_review(payload: ReviewRequest) -> dict:
        return get_review_store().add(
            prediction_id=payload.prediction_id,
            decision=payload.decision,
            predicted_class_id=payload.predicted_class_id,
            corrected_class_id=payload.corrected_class_id,
            note=payload.note,
            image_consent=payload.image_consent,
        )

    @app.get("/api/v1/reviews/summary")
    def review_summary() -> dict:
        return get_review_store().summary()

    @app.get("/", include_in_schema=False)
    def web_app() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "index.html")

    @app.get("/app.js", include_in_schema=False)
    def web_script() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "app.js", media_type="text/javascript")

    @app.get("/styles.css", include_in_schema=False)
    def web_styles() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "styles.css", media_type="text/css")

    return app


app = create_app()
