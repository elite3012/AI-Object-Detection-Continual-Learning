from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.training import TrainingOverrides, load_training_config, run_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PestNet-S on the audited IP102 subset")
    parser.add_argument("--config", type=Path, default=Path("configs/train/pestnet_s.yaml"))
    parser.add_argument("--max-epochs", type=int, help="Override epochs for smoke runs")
    parser.add_argument(
        "--limit-train-per-class",
        type=int,
        help="Use a small deterministic slice per class for smoke runs",
    )
    parser.add_argument(
        "--limit-val-per-class",
        type=int,
        help="Use a small deterministic validation slice per class for smoke runs",
    )
    parser.add_argument("--device", help="Override device, for example cpu or cuda")
    parser.add_argument(
        "--batch-size", type=int, help="Override training and validation batch size"
    )
    parser.add_argument("--num-workers", type=int, help="Override PyTorch DataLoader workers")
    parser.add_argument("--bundle-dir", type=Path, help="Override exported model-bundle directory")
    parser.add_argument("--progress", action="store_true", help="Print one JSON line per epoch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        load_training_config(args.config),
        overrides=TrainingOverrides(
            max_epochs=args.max_epochs,
            limit_train_per_class=args.limit_train_per_class,
            limit_val_per_class=args.limit_val_per_class,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bundle_dir=args.bundle_dir,
            log_progress=args.progress,
        ),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
