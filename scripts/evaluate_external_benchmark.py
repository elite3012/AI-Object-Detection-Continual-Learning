from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.evaluation import evaluate_external_benchmark  # noqa: E402


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a bundle on external licensed images")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/models/pestnet_s_latest"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/external_benchmark/manifest.json"),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    _configure_stdout()
    args = parse_args()
    result = evaluate_external_benchmark(
        bundle_dir=args.bundle_dir,
        manifest_path=args.manifest,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
