from __future__ import annotations

from pathlib import Path

from pestscope.evaluation import CalibrationPolicy, select_thresholds
from pestscope.inference.service import InferenceService
from pestscope.modeling import build_model, count_parameters
from pestscope.training.bundle import write_bundle_thresholds, write_model_bundle
from pestscope.training.transforms import DEFAULT_MEAN, DEFAULT_STD


def test_select_thresholds_balances_acceptance_and_near_ood() -> None:
    id_scores = [
        {"confidence": 0.91, "correct": True},
        {"confidence": 0.82, "correct": True},
        {"confidence": 0.76, "correct": False},
        {"confidence": 0.42, "correct": True},
    ]
    ood_scores = [
        {"confidence": 0.22},
        {"confidence": 0.31},
        {"confidence": 0.36},
    ]

    thresholds = select_thresholds(
        id_scores,
        ood_scores,
        policy=CalibrationPolicy(target_accept_precision=0.9, min_accept_coverage=0.25),
    )

    assert thresholds["accepted"] >= thresholds["uncertain"]
    assert thresholds["selection"]["met_target"] is True
    assert thresholds["selection"]["accepted_precision"] >= 0.9


def test_select_thresholds_refuses_acceptance_when_precision_target_fails() -> None:
    id_scores = [
        {"confidence": 0.91, "correct": False},
        {"confidence": 0.82, "correct": False},
        {"confidence": 0.45, "correct": True},
    ]

    thresholds = select_thresholds(
        id_scores,
        [],
        policy=CalibrationPolicy(target_accept_precision=0.8, min_accept_coverage=0.2),
    )

    assert thresholds["selection"]["met_target"] is False
    assert thresholds["selection"]["conservative_no_acceptance"] is True
    assert thresholds["accepted"] > 0.91


def test_inference_service_uses_bundle_thresholds_when_env_does_not_override(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    model = build_model("pestnet_s", num_classes=2, width=8, dropout=0.0)
    write_model_bundle(
        bundle_dir=bundle_dir,
        model=model,
        metadata={
            "schema_version": 1,
            "run_id": "threshold-fixture",
            "model": {
                "name": "pestnet_s",
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
        accepted=0.64,
        uncertain=0.21,
        calibration={"source": "test"},
    )

    service = InferenceService.from_bundle(
        bundle_dir,
        device="cpu",
        accept_threshold=None,
        uncertain_threshold=None,
    )

    assert service.accept_threshold == 0.64
    assert service.uncertain_threshold == 0.21
