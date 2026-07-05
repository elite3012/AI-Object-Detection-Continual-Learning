from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.evaluation import CalibrationPolicy, evaluate_bundle  # noqa: E402
from pestscope.training import load_training_config  # noqa: E402


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PestNet model bundle and calibrate confidence thresholds."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train/pestnet_s.yaml"))
    parser.add_argument("--bundle-dir", type=Path, help="Model bundle to evaluate")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-id-per-class", type=int)
    parser.add_argument("--limit-ood-per-class", type=int)
    parser.add_argument("--target-accept-precision", type=float, default=0.70)
    parser.add_argument("--min-accept-coverage", type=float, default=0.05)
    parser.add_argument("--ood-reject-quantile", type=float, default=0.95)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/pestnet_s_latest_eval.json"),
    )
    parser.add_argument(
        "--write-thresholds",
        action="store_true",
        help="Write selected thresholds and calibration summary into metadata.json",
    )
    return parser.parse_args()


def main() -> int:
    _configure_stdout()
    args = parse_args()
    config = load_training_config(args.config)
    bundle_dir = args.bundle_dir or config.outputs.bundle_dir
    result = evaluate_bundle(
        bundle_dir=bundle_dir,
        dataset_root=config.data.dataset_root,
        manifest_path=config.data.manifest_path,
        selected_class_ids=config.data.selected_class_ids,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        id_limit_per_class=args.limit_id_per_class,
        ood_limit_per_class=args.limit_ood_per_class,
        policy=CalibrationPolicy(
            target_accept_precision=args.target_accept_precision,
            min_accept_coverage=args.min_accept_coverage,
            ood_reject_quantile=args.ood_reject_quantile,
        ),
        output_path=args.output,
        write_thresholds=args.write_thresholds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
