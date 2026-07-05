from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.evaluation import CalibrationPolicy, evaluate_bundle  # noqa: E402
from pestscope.training import load_training_config  # noqa: E402
from pestscope.training.bundle import load_model_bundle  # noqa: E402


def _read_suite(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment suite must be a mapping: {path}")
    return raw


def _metric_row(name: str, evaluation: dict, metadata: dict) -> dict:
    classification = evaluation["classification"]
    threshold = evaluation["thresholds"]
    decision = evaluation["decision_summary"]
    return {
        "name": name,
        "run_id": metadata.get("run_id"),
        "architecture": metadata["model"]["name"],
        "parameters": metadata["model"]["parameter_count"],
        "epochs": metadata.get("training", {}).get("epochs_run"),
        "top1_accuracy": classification["top1_accuracy"],
        "top3_accuracy": classification["top3_accuracy"],
        "macro_f1": classification["macro_f1"],
        "balanced_accuracy": classification["balanced_accuracy"],
        "accepted_threshold": threshold["accepted"],
        "uncertain_threshold": threshold["uncertain"],
        "accepted_precision": decision["id_accepted_precision"],
        "accepted_coverage": decision["id_coverage"],
        "near_ood_accepted_rate": decision["near_ood_accepted_rate"],
        "near_ood_unsupported_rate": decision["near_ood_unsupported_rate"],
    }


def _write_markdown(path: Path, rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["macro_f1"], reverse=True)
    lines = [
        "# Section 6 Model Comparison",
        "",
        "| Model | Params | Top-1 | Top-3 | Macro-F1 | Accepted precision | "
        "Accepted coverage | Near-OOD accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            "| {name} | {params:,} | {top1:.4f} | {top3:.4f} | {f1:.4f} | "
            "{precision:.4f} | {coverage:.4f} | {ood:.4f} |".format(
                name=row["name"],
                params=row["parameters"],
                top1=row["top1_accuracy"],
                top3=row["top3_accuracy"],
                f1=row["macro_f1"],
                precision=row["accepted_precision"],
                coverage=row["accepted_coverage"],
                ood=row["near_ood_accepted_rate"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(path: Path) -> dict:
    suite = _read_suite(path)
    rows = []
    evaluations = {}
    for experiment in suite["experiments"]:
        config = load_training_config(Path(experiment["config"]))
        bundle_dir = Path(experiment.get("bundle_dir") or config.outputs.bundle_dir)
        _, metadata = load_model_bundle(bundle_dir, device="cpu")
        evaluation = evaluate_bundle(
            bundle_dir=bundle_dir,
            dataset_root=config.data.dataset_root,
            manifest_path=config.data.manifest_path,
            selected_class_ids=config.data.selected_class_ids,
            split="val",
            batch_size=int(suite.get("batch_size", 64)),
            device=str(suite.get("device", "cpu")),
            id_limit_per_class=suite.get("limit_id_per_class"),
            ood_limit_per_class=suite.get("limit_ood_per_class"),
            policy=CalibrationPolicy(),
        )
        rows.append(_metric_row(str(experiment["name"]), evaluation, metadata))
        evaluations[str(experiment["name"])] = evaluation

    result = {
        "schema_version": 1,
        "suite": suite["name"],
        "rows": sorted(rows, key=lambda row: row["macro_f1"], reverse=True),
        "evaluations": evaluations,
    }
    outputs = suite.get("outputs", {})
    output_json = Path(outputs.get("comparison_json", "artifacts/evaluation/model_comparison.json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(
        Path(outputs.get("comparison_markdown", "artifacts/evaluation/model_comparison.md")),
        result["rows"],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained PestScope model bundles")
    parser.add_argument("--suite", type=Path, default=Path("configs/experiments/section6.yaml"))
    return parser.parse_args()


def main() -> int:
    result = run_suite(parse_args().suite)
    print(json.dumps({"suite": result["suite"], "rows": result["rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
