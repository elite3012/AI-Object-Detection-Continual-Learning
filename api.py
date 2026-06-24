from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from models.adaptive_service import AdaptiveVisionService, NoClassesError, Prediction
from models.config import Settings
from models.demo_catalog import (
    DEMO_SAMPLES,
    SAMPLE_BY_ID,
    demo_image_bytes,
    known_demo_samples,
    render_demo_image,
    training_images,
)
from models.drift import DriftMonitor
from models.embeddings import CLIPImageEmbedder
from models.prototype_memory import PrototypeMemory
from models.visual_embeddings import VisualFeatureEmbedder


def build_service(settings: Settings) -> AdaptiveVisionService:
    encoder = os.getenv("VISION_ENCODER", "clip").lower()
    embedder = (
        VisualFeatureEmbedder()
        if encoder == "visual"
        else CLIPImageEmbedder(settings.model_name, settings.device)
    )
    return AdaptiveVisionService(
        embedder=embedder,
        memory=PrototypeMemory(settings.state_path),
        monitor=DriftMonitor(settings.drift_window_size),
        confidence_threshold=settings.confidence_threshold,
    )


def create_app(service: AdaptiveVisionService | None = None) -> FastAPI:
    settings = Settings.from_env()
    app = FastAPI(
        title="Adaptive Vision Service",
        version="1.0.0",
        description="Few-shot image classification with persistent prototype memory.",
    )
    app.state.service = service
    app.state.demo_lock = Lock()

    def get_service() -> AdaptiveVisionService:
        if app.state.service is None:
            app.state.service = build_service(settings)
        return app.state.service

    async def read_image(file: UploadFile) -> Image.Image:
        limit = settings.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {settings.max_upload_mb} MB",
            )
        try:
            image = Image.open(BytesIO(content))
            image.load()
            return image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail="File is not a valid image") from exc

    def prediction_payload(result: Prediction) -> dict:
        return {
            "label": result.label,
            "is_unknown": result.is_unknown,
            "threshold": result.threshold,
            "matches": [
                {
                    "label": match.label,
                    "similarity": match.similarity,
                    "examples": match.examples,
                }
                for match in result.matches
            ],
        }

    def demo_state() -> dict:
        classes = {item["label"] for item in get_service().classes()}
        required = {sample.label for sample in known_demo_samples()}
        ready = required.issubset(classes)
        return {
            "status": "ready" if ready else "setup_required",
            "ready": ready,
            "samples": [
                {
                    **sample.to_dict(),
                    "image_url": f"/v1/demo/samples/{sample.id}/image",
                }
                for sample in DEMO_SAMPLES
            ],
        }

    def bootstrap_demo_data() -> dict:
        with app.state.demo_lock:
            current = get_service()
            labels = {item["label"] for item in current.classes()}
            for sample in known_demo_samples():
                if sample.label not in labels:
                    current.teach(sample.label, training_images(sample.id))
            return demo_state()

    @app.get("/health")
    def health() -> dict:
        current = get_service()
        demo = demo_state()
        return {
            "status": "ok",
            "model": getattr(current.embedder, "model_name", type(current.embedder).__name__),
            "model_loaded": bool(getattr(current.embedder, "is_loaded", True)),
            "classes": len(current.classes()),
            "demo_ready": demo["ready"],
        }

    @app.get("/v1/classes")
    def list_classes() -> dict:
        return {"classes": get_service().classes()}

    @app.post("/v1/classes/{label}/examples", status_code=status.HTTP_201_CREATED)
    async def teach_class(label: str, files: list[UploadFile] = File(...)) -> dict:
        try:
            images = [await read_image(file) for file in files]
            return get_service().teach(label, images)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/v1/classes/{label}")
    def delete_class(label: str) -> dict:
        try:
            if not get_service().delete_class(label):
                raise HTTPException(status_code=404, detail="Class not found")
            return {"deleted": label}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/predict")
    async def predict(
        file: UploadFile = File(...),
        top_k: int = Query(default=3, ge=1, le=20),
    ) -> dict:
        try:
            result = get_service().predict(await read_image(file), top_k=top_k)
        except NoClassesError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return prediction_payload(result)

    @app.post("/v1/feedback/{label}")
    async def feedback(label: str, file: UploadFile = File(...)) -> dict:
        try:
            return get_service().feedback(label, await read_image(file))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/metrics")
    def metrics() -> dict:
        return get_service().metrics()

    @app.get("/v1/demo")
    def get_demo() -> dict:
        return demo_state()

    @app.post("/v1/demo/bootstrap")
    def bootstrap_demo() -> dict:
        try:
            return bootstrap_demo_data()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Demo setup failed: {exc}",
            ) from exc

    @app.get("/v1/demo/samples/{sample_id}/image", response_class=Response)
    def demo_image(sample_id: str) -> Response:
        if sample_id not in SAMPLE_BY_ID:
            raise HTTPException(status_code=404, detail="Demo sample not found")
        return Response(
            content=demo_image_bytes(sample_id),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/v1/demo/samples/{sample_id}/predict")
    def predict_demo_sample(
        sample_id: str,
        top_k: int = Query(default=3, ge=1, le=20),
    ) -> dict:
        if sample_id not in SAMPLE_BY_ID:
            raise HTTPException(status_code=404, detail="Demo sample not found")
        if not demo_state()["ready"]:
            raise HTTPException(status_code=409, detail="Bootstrap demo data first")
        result = get_service().predict(render_demo_image(sample_id), top_k=top_k)
        return prediction_payload(result)

    project_root = Path(__file__).resolve().parent

    @app.get("/", include_in_schema=False)
    def web_app() -> FileResponse:
        return FileResponse(project_root / "index.html")

    @app.get("/app.js", include_in_schema=False)
    def web_script() -> FileResponse:
        return FileResponse(project_root / "app.js", media_type="text/javascript")

    @app.get("/styles.css", include_in_schema=False)
    def web_styles() -> FileResponse:
        return FileResponse(project_root / "styles.css", media_type="text/css")

    return app


app = create_app()
