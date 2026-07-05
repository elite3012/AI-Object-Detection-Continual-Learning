from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from pestscope.evaluation import evaluate_external_benchmark
from pestscope.modeling import build_model, count_parameters
from pestscope.training.bundle import write_bundle_thresholds, write_model_bundle
from pestscope.training.transforms import DEFAULT_MEAN, DEFAULT_STD


def test_evaluate_external_benchmark_scores_local_manifest(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 64), (110, 150, 90)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "class_id": 1,
                        "canonical_name": "Cnaphalocrocis medinalis",
                        "common_name_vi": "Sau cuon la nho",
                        "provider": "fixture",
                        "license": "test",
                        "source_url": "https://example.test/sample.jpg",
                        "status": "ok",
                        "image_path": str(image_path),
                        "error": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "bundle"
    model = build_model("simple_cnn", num_classes=2, width=8, dropout=0.0)
    write_model_bundle(
        bundle_dir=bundle_dir,
        model=model,
        metadata={
            "schema_version": 1,
            "run_id": "external-fixture",
            "model": {
                "name": "simple_cnn",
                "width": 8,
                "dropout": 0.0,
                "num_classes": 2,
                "parameter_count": count_parameters(model),
            },
            "dataset": {},
            "preprocessing": {
                "image_size": 32,
                "mean": list(DEFAULT_MEAN),
                "std": list(DEFAULT_STD),
            },
            "classes": [
                {
                    "index": 0,
                    "ip102_id": 1,
                    "dataset_label": "one",
                    "canonical_name": "One",
                    "common_name_en": "one",
                    "common_name_vi": "Mot",
                    "stratum": "head",
                },
                {
                    "index": 1,
                    "ip102_id": 2,
                    "dataset_label": "two",
                    "canonical_name": "Two",
                    "common_name_en": "two",
                    "common_name_vi": "Hai",
                    "stratum": "tail",
                },
            ],
        },
        metrics={"schema_version": 1},
    )
    write_bundle_thresholds(
        bundle_dir,
        accepted=0.99,
        uncertain=0.50,
        calibration={"source": "test"},
    )

    result = evaluate_external_benchmark(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        device="cpu",
    )

    assert result["summary"]["evaluated"] == 1
    assert result["records"][0]["evaluated"] is True
    assert "top3_class_ids" in result["records"][0]
    assert (tmp_path / "evaluation.json").is_file()
