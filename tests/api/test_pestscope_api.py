from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api import create_app
from pestscope.inference.config import InferenceSettings
from pestscope.inference.reviews import ReviewStore
from pestscope.inference.service import InferenceService
from pestscope.modeling import build_model, count_parameters
from pestscope.training.bundle import write_model_bundle
from pestscope.training.transforms import DEFAULT_MEAN, DEFAULT_STD


def _png_bytes(color: tuple[int, int, int] = (180, 60, 40)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (40, 36), color).save(output, format="PNG")
    return output.getvalue()


def _review_config(path: Path) -> None:
    path.write_text(
        """
schema_version: 1
status: fixture
classes:
  - ip102_id: 1
    dataset_label: rice leaf roller
    canonical_name: Cnaphalocrocis medinalis
    common_name_en: rice leaf folder
    common_name_vi: Sau cuon la nho
    stratum: head
    external_source: {provider: Fixture, license: CC0, url: https://example.test/one.jpg}
  - ip102_id: 8
    dataset_label: brown plant hopper
    canonical_name: Nilaparvata lugens
    common_name_en: brown planthopper
    common_name_vi: Ray nau
    stratum: head
    external_source: {provider: Fixture, license: CC0, url: https://example.test/eight.jpg}
  - ip102_id: 40
    dataset_label: beet army worm
    canonical_name: Spodoptera exigua
    common_name_en: beet armyworm
    common_name_vi: Sau xanh da lang
    stratum: head
    external_source: {provider: Fixture, license: CC0, url: https://example.test/forty.jpg}
  - ip102_id: 77
    dataset_label: Icerya purchasi Maskell
    canonical_name: Icerya purchasi
    common_name_en: cottony cushion scale
    common_name_vi: Rep sap bong
    stratum: middle
    external_source: {provider: Fixture, license: CC0, url: https://example.test/seventy-seven.jpg}
""".strip(),
        encoding="utf-8",
    )


def _bundle(path: Path) -> None:
    model = build_model("pestnet_s", num_classes=4, width=8, dropout=0.0)
    classes = [
        {
            "index": index,
            "ip102_id": class_id,
            "dataset_label": label,
            "canonical_name": canonical,
            "common_name_en": english,
            "common_name_vi": vietnamese,
            "stratum": stratum,
        }
        for index, class_id, label, canonical, english, vietnamese, stratum in [
            (
                0,
                1,
                "rice leaf roller",
                "Cnaphalocrocis medinalis",
                "rice leaf folder",
                "Sau cuon la nho",
                "head",
            ),
            (
                1,
                8,
                "brown plant hopper",
                "Nilaparvata lugens",
                "brown planthopper",
                "Ray nau",
                "head",
            ),
            (
                2,
                40,
                "beet army worm",
                "Spodoptera exigua",
                "beet armyworm",
                "Sau xanh da lang",
                "head",
            ),
            (
                3,
                77,
                "Icerya purchasi Maskell",
                "Icerya purchasi",
                "cottony cushion scale",
                "Rep sap bong",
                "middle",
            ),
        ]
    ]
    write_model_bundle(
        bundle_dir=path,
        model=model,
        metadata={
            "schema_version": 1,
            "created_at": "2026-06-24T00:00:00+00:00",
            "run_id": "fixture-run",
            "dataset": {"manifest_sha256": "abc123"},
            "model": {
                "name": "pestnet_s",
                "width": 8,
                "dropout": 0.0,
                "num_classes": 4,
                "parameter_count": count_parameters(model),
            },
            "preprocessing": {
                "image_size": 32,
                "mean": list(DEFAULT_MEAN),
                "std": list(DEFAULT_STD),
            },
            "classes": classes,
        },
        metrics={"schema_version": 1, "run_id": "fixture-run"},
    )


def _settings(tmp_path: Path) -> InferenceSettings:
    review = tmp_path / "review.yaml"
    _review_config(review)
    bundle = tmp_path / "bundle"
    _bundle(bundle)
    return InferenceSettings(
        model_bundle=bundle,
        class_review=review,
        device="cpu",
        fetch_demo_images=False,
        demo_cache_dir=tmp_path / "demo-cache",
        review_db=tmp_path / "reviews.sqlite3",
    )


def test_prediction_api_uses_model_bundle_and_stores_review(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(
        create_app(
            service=service,
            settings=settings,
            review_store=ReviewStore(settings.review_db),
        )
    )

    ready = client.get("/api/v1/health/ready")
    model = client.get("/api/v1/model")
    examples = client.get("/api/v1/examples")
    prediction = client.post(
        "/api/v1/predictions",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert ready.status_code == 200
    assert ready.json()["model_version"] == "fixture-run"
    assert model.json()["model"]["num_classes"] == 4
    assert len(examples.json()["examples"]) == 4
    assert prediction.status_code == 200
    assert prediction.json()["decision"] == "accepted"
    assert len(prediction.json()["top_k"]) == 3

    review = client.post(
        "/api/v1/reviews",
        json={
            "prediction_id": prediction.json()["prediction_id"],
            "decision": prediction.json()["decision"],
            "predicted_class_id": prediction.json()["top_k"][0]["class_id"],
            "corrected_class_id": 8,
            "note": "fixture correction",
            "image_consent": True,
        },
    )
    summary = client.get("/api/v1/reviews/summary")

    assert review.status_code == 201
    assert review.json()["image_retained"] is False
    assert summary.json()["review_count"] == 1


def test_missing_bundle_can_boot_demo_model(tmp_path: Path) -> None:
    review = tmp_path / "review.yaml"
    _review_config(review)
    settings = InferenceSettings(
        model_bundle=tmp_path / "missing" / "pestnet_s_latest",
        class_review=review,
        device="cpu",
        allow_demo_model=True,
        fetch_demo_images=False,
        demo_cache_dir=tmp_path / "demo-cache",
        review_db=tmp_path / "reviews.sqlite3",
    )
    client = TestClient(create_app(settings=settings))

    ready = client.get("/api/v1/health/ready")
    example_image = client.get("/api/v1/examples/class-1/image")

    assert ready.status_code == 200
    assert ready.json()["demo_model"] is True
    assert example_image.status_code == 200
    assert example_image.headers["content-type"] == "image/jpeg"
