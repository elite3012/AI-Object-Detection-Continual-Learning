from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pestscope.data.config import load_data_config  # noqa: E402
from pestscope.data.pipeline import run_data_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the IP102 manifest, leakage audit, EDA report, and shortlist."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/ip102_subset.yaml"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override dataset.root from the configuration",
    )
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Re-run the audit and shortlist without scanning the image files again",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_data_config(args.config, args.dataset_root)
    if not config.dataset_root.is_dir():
        raise SystemExit(
            f"Dataset root does not exist: {config.dataset_root}. "
            "Run scripts/download_ip102.py --list-sources first."
        )
    result = run_data_pipeline(config, rebuild_manifest=not args.reuse_manifest)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
