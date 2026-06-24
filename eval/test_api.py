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


def test_first_run_demo_workflow_and_web_app() -> None:
    service = AdaptiveVisionService(
        MeanColorEmbedder(),
        PrototypeMemory(),
        confidence_threshold=0.8,
    )
    client = TestClient(create_app(service))

    web_app = client.get("/")
    initial = client.get("/v1/demo")
    bootstrap = client.post("/v1/demo/bootstrap")
    image = client.get("/v1/demo/samples/connector-pass/image")
    prediction = client.post("/v1/demo/samples/connector-pass/predict")

    assert web_app.status_code == 200
    assert "SignalLens" in web_app.text
    assert initial.json()["ready"] is False
    assert len(initial.json()["samples"]) == 4
    assert bootstrap.status_code == 200
    assert bootstrap.json()["ready"] is True
    assert len(service.classes()) == 3
    assert image.headers["content-type"] == "image/png"
    assert prediction.status_code == 200
    assert prediction.json()["label"] == "connector-pass"
