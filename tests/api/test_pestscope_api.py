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
            "git": {"commit": "abc123def456", "dirty": False},
            "dataset": {
                "manifest_sha256": "abc123",
                "selected_class_ids": [1, 8, 40, 77],
                "train_records": 40,
                "val_records": 12,
            },
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
            "training": {
                "seed": 2026,
                "epochs_requested": 2,
                "epochs_run": 2,
                "batch_size": 4,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "device": "cpu",
                "class_strategy": "weighted_loss",
            },
            "artifact": {
                "model_file": "model.pt",
                "model_sha256": "0123456789abcdef",
                "metrics_file": "metrics.json",
            },
            "classes": classes,
        },
        metrics={
            "schema_version": 1,
            "run_id": "fixture-run",
            "history": [
                {
                    "epoch": 1,
                    "train_loss": 1.4,
                    "train_macro_f1": 0.2,
                    "train_top1_accuracy": 0.3,
                    "val_loss": 1.3,
                    "val_macro_f1": 0.24,
                    "val_top1_accuracy": 0.35,
                },
                {
                    "epoch": 2,
                    "train_loss": 1.1,
                    "train_macro_f1": 0.36,
                    "train_top1_accuracy": 0.46,
                    "val_loss": 1.0,
                    "val_macro_f1": 0.42,
                    "val_top1_accuracy": 0.5,
                },
            ],
            "best_validation": {
                "epoch": 2,
                "loss": 1.0,
                "samples": 12,
                "top1_accuracy": 0.5,
                "top3_accuracy": 0.75,
                "macro_f1": 0.42,
                "balanced_accuracy": 0.44,
                "confusion_matrix": [
                    [3, 0, 0, 0],
                    [1, 2, 0, 0],
                    [0, 1, 2, 0],
                    [0, 0, 1, 2],
                ],
                "per_class": [
                    {"index": 0, "support": 3, "precision": 0.75, "recall": 1.0, "f1": 0.86},
                    {"index": 1, "support": 3, "precision": 0.67, "recall": 0.67, "f1": 0.67},
                    {"index": 2, "support": 3, "precision": 0.67, "recall": 0.67, "f1": 0.67},
                    {"index": 3, "support": 3, "precision": 1.0, "recall": 0.67, "f1": 0.8},
                ],
            },
        },
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
        demo_dataset_root=tmp_path / "missing-dataset",
        demo_manifest=tmp_path / "missing-manifest.csv",
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
    assert model.json()["metrics"]["best_validation"]["macro_f1"] == 0.42
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


def test_experiment_evidence_endpoint_exposes_current_bundle(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/experiments/current")
    payload = response.json()

    assert response.status_code == 200
    assert payload["run"]["run_id"] == "fixture-run"
    assert payload["run"]["checkpoint_file"] == "model.pt"
    assert len(payload["run"]["checkpoint_sha256_short"]) == 12
    assert payload["training"]["epochs_run"] == 2
    assert payload["augmentation"]["crop_scale"] == [0.82, 1.0]
    assert len(payload["curves"]) == 2
    assert payload["curves"][-1]["val_macro_f1"] == 0.42
    assert payload["confusion"]["matrix"][1][0] == 1
    assert payload["confusion"]["top_pairs"][0]["count"] == 1
    assert len(payload["class_distribution"]) == 4
    assert payload["failure_analysis"]["weakest_classes"][0]["name"] == "brown planthopper"
    assert payload["failure_analysis"]["confusion_drivers"][0]["actual"] == "brown planthopper"
    assert payload["failure_analysis"]["root_causes"]
    assert payload["failure_analysis"]["improvement_steps"]
    assert payload["reproducibility"]["seed"] == 2026
    assert payload["reproducibility"]["config_path"] == "configs\\train\\pestnet_s.yaml"
    assert "scripts\\train_pestnet.py" in payload["reproducibility"]["commands"]["train"]
    assert (
        "scripts\\evaluate_pestnet_bundle.py" in payload["reproducibility"]["commands"]["evaluate"]
    )
    assert any(
        item["label"] == "Bundle metadata" for item in payload["reproducibility"]["artifacts"]
    )


def test_example_stem_activations_are_real_feature_maps(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/examples/class-1/stem-activations?channel_count=3")
    payload = response.json()

    assert response.status_code == 200
    assert payload["layer"] == "features.0"
    assert payload["input_shape"] == [3, 32, 32]
    assert payload["output_shape"] == [8, 16, 16]
    assert len(payload["channels"]) == 3
    assert payload["channels"][0]["image"].startswith("data:image/png;base64,")


def test_example_residual32_activations_compare_before_and_after(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/examples/class-1/residual32-activations?channel_count=2")
    payload = response.json()

    assert response.status_code == 200
    assert payload["layer"] == "features.1"
    assert payload["shortcut"] == "identity"
    assert payload["input_shape"] == [8, 16, 16]
    assert payload["output_shape"] == [8, 16, 16]
    assert len(payload["channels"]) == 2
    assert payload["channels"][0]["before_image"].startswith("data:image/png;base64,")
    assert payload["channels"][0]["after_image"].startswith("data:image/png;base64,")


def test_example_residual64_activations_show_projection_shortcut(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/examples/class-1/residual64-activations?channel_count=2")
    payload = response.json()

    assert response.status_code == 200
    assert payload["layer"] == "features.2"
    assert payload["shortcut"] == "projection 1x1 stride=2"
    assert payload["input_shape"] == [8, 16, 16]
    assert payload["branch_shape"] == [16, 8, 8]
    assert payload["shortcut_shape"] == [16, 8, 8]
    assert payload["output_shape"] == [16, 8, 8]
    assert len(payload["channels"]) == 2
    assert payload["channels"][0]["branch_image"].startswith("data:image/png;base64,")
    assert payload["channels"][0]["shortcut_image"].startswith("data:image/png;base64,")
    assert payload["channels"][0]["output_image"].startswith("data:image/png;base64,")


def test_example_residual128_activations_show_deeper_projection(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/examples/class-1/residual128-activations?channel_count=2")
    payload = response.json()

    assert response.status_code == 200
    assert payload["layer"] == "features.4"
    assert payload["shortcut"] == "projection 1x1 stride=2"
    assert payload["input_shape"] == [16, 8, 8]
    assert payload["branch_shape"] == [32, 4, 4]
    assert payload["shortcut_shape"] == [32, 4, 4]
    assert payload["output_shape"] == [32, 4, 4]
    assert len(payload["channels"]) == 2
    assert payload["channels"][0]["branch_image"].startswith("data:image/png;base64,")
    assert payload["channels"][0]["shortcut_image"].startswith("data:image/png;base64,")
    assert payload["channels"][0]["output_image"].startswith("data:image/png;base64,")


def test_example_attention_activations_show_channel_gates(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/examples/class-1/attention-activations?channel_count=3")
    payload = response.json()

    assert response.status_code == 200
    assert payload["layer"] == "features.6.attention"
    assert payload["input_shape"] == [32, 4, 4]
    assert payload["branch_shape"] == [64, 2, 2]
    assert payload["gate_shape"] == [64, 1, 1]
    assert payload["output_shape"] == [64, 2, 2]
    assert 0 <= payload["gate_summary"]["min"] <= payload["gate_summary"]["max"] <= 1
    assert len(payload["channels"]) == 3
    assert payload["channels"][0]["before_image"].startswith("data:image/png;base64,")
    assert payload["channels"][0]["after_image"].startswith("data:image/png;base64,")


def test_example_global_pool_activations_show_vector_summary(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.0,
        uncertain_threshold=0.0,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/examples/class-1/global-pool-activations?channel_count=4")
    payload = response.json()

    assert response.status_code == 200
    assert payload["layer"] == "head.0"
    assert payload["input_shape"] == [64, 2, 2]
    assert payload["output_shape"] == [64]
    assert payload["pooling"] == "mean over H x W"
    assert len(payload["channels"]) == 4
    assert payload["channels"][0]["image"].startswith("data:image/png;base64,")
    assert payload["channels"][0]["pooled_value"] == payload["channels"][0]["mean"]


def test_example_decision_gate_shows_softmax_thresholds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    service = InferenceService.from_bundle(
        settings.model_bundle,
        device="cpu",
        accept_threshold=0.6,
        uncertain_threshold=0.2,
    )
    client = TestClient(create_app(service=service, settings=settings))

    response = client.get("/api/v1/examples/class-1/decision-gate?top_k=3")
    payload = response.json()

    assert response.status_code == 200
    assert payload["layer"] == "head.3 + softmax"
    assert payload["vector_shape"] == [64]
    assert payload["logits_shape"] == [4]
    assert payload["thresholds"] == {"accepted": 0.6, "uncertain": 0.2}
    assert payload["decision"] in {"accepted", "uncertain", "unsupported"}
    assert len(payload["top_k"]) == 3
    assert "logit" in payload["top_k"][0]
    assert payload["confidence"] == payload["top_k"][0]["confidence"]


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
        demo_dataset_root=tmp_path / "missing-dataset",
        demo_manifest=tmp_path / "missing-manifest.csv",
        review_db=tmp_path / "reviews.sqlite3",
    )
    client = TestClient(create_app(settings=settings))

    ready = client.get("/api/v1/health/ready")
    example_image = client.get("/api/v1/examples/class-1/image")

    assert ready.status_code == 200
    assert ready.json()["demo_model"] is True
    assert example_image.status_code == 200
    assert example_image.headers["content-type"] == "image/jpeg"


def test_examples_use_local_ip102_manifest_images_when_available(tmp_path: Path) -> None:
    review = tmp_path / "review.yaml"
    review.write_text(
        """
schema_version: 1
status: fixture
classes:
  - ip102_id: 72
    dataset_label: greenhouse whitefly
    canonical_name: Trialeurodes vaporariorum
    common_name_en: greenhouse whitefly
    common_name_vi: Bo phan trang nha kinh
    stratum: middle
    external_source: {provider: Fixture, license: CC0, url: https://example.test/whitefly.jpg}
""".strip(),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "ip102"
    image_path = dataset_root / "images" / "sample-72.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(_png_bytes(color=(12, 180, 90)))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_id,path,split,class_id,class_name,status\n"
        "sample-72,images/sample-72.png,test,72,greenhouse whitefly,ok\n",
        encoding="utf-8",
    )
    settings = InferenceSettings(
        class_review=review,
        device="cpu",
        fetch_demo_images=False,
        demo_cache_dir=tmp_path / "demo-cache",
        demo_dataset_root=dataset_root,
        demo_manifest=manifest,
        review_db=tmp_path / "reviews.sqlite3",
    )
    client = TestClient(create_app(settings=settings))

    examples = client.get("/api/v1/examples").json()["examples"]
    example_image = client.get("/api/v1/examples/class-72/image")

    assert examples[0]["image_kind"] == "dataset"
    assert example_image.status_code == 200
    assert example_image.headers["content-type"] == "image/jpeg"
