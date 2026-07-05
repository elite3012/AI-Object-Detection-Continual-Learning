from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.evaluation import build_external_benchmark  # noqa: E402


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the licensed external-image benchmark")
    parser.add_argument(
        "--class-review",
        type=Path,
        default=Path("configs/data/ip102_class_review.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/external_benchmark"),
    )
    return parser.parse_args()


def main() -> int:
    _configure_stdout()
    args = parse_args()
    manifest = build_external_benchmark(
        class_review_path=args.class_review,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
