from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from PIL import Image, UnidentifiedImageError

from models.adaptive_service import AdaptiveVisionService, NoClassesError
from models.config import Settings
from models.drift import DriftMonitor
from models.embeddings import CLIPImageEmbedder
from models.prototype_memory import PrototypeMemory


def build_service(settings: Settings) -> AdaptiveVisionService:
    return AdaptiveVisionService(
        embedder=CLIPImageEmbedder(settings.model_name, settings.device),
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

    @app.get("/health")
    def health() -> dict:
        current = get_service()
        return {
            "status": "ok",
            "model": getattr(current.embedder, "model_name", type(current.embedder).__name__),
            "model_loaded": bool(getattr(current.embedder, "is_loaded", True)),
            "classes": len(current.classes()),
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

    @app.post("/v1/feedback/{label}")
    async def feedback(label: str, file: UploadFile = File(...)) -> dict:
        try:
            return get_service().feedback(label, await read_image(file))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/metrics")
    def metrics() -> dict:
        return get_service().metrics()

    return app


app = create_app()
