from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api import create_app
from eval.test_service import MeanColorEmbedder
from models.adaptive_service import AdaptiveVisionService
from models.prototype_memory import PrototypeMemory


def png_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 16), color).save(output, format="PNG")
    return output.getvalue()


def test_teach_and_predict_api() -> None:
    service = AdaptiveVisionService(
        MeanColorEmbedder(),
        PrototypeMemory(),
        confidence_threshold=0.8,
    )
    client = TestClient(create_app(service))

    teach = client.post(
        "/v1/classes/red/examples",
        files=[("files", ("red.png", png_bytes((255, 0, 0)), "image/png"))],
    )
    predict = client.post(
        "/v1/predict?top_k=1",
        files={"file": ("query.png", png_bytes((250, 5, 0)), "image/png")},
    )

    assert teach.status_code == 201
    assert predict.status_code == 200
    assert predict.json()["label"] == "red"
    assert predict.json()["matches"][0]["similarity"] > 0.99
