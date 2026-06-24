from __future__ import annotations

import argparse
import json
from pathlib import Path

from pestscope.data.acquisition import (
    ACADEMIC_USE_NOTICE,
    OFFICIAL_REPOSITORY,
    OFFICIAL_SOURCES,
    AcquisitionError,
    acquire_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire and safely extract an IP102 archive after reviewing its terms."
    )
    parser.add_argument("--list-sources", action="store_true", help="Print official sources")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Previously downloaded archive")
    source.add_argument("--url", help="Direct archive URL supplied by the user")
    parser.add_argument("--destination", type=Path, default=Path("data/raw/ip102"))
    parser.add_argument("--sha256", help="Expected archive SHA-256")
    parser.add_argument("--accept-academic-use", action="store_true")
    parser.add_argument("--delete-downloaded-archive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_sources or (args.archive is None and args.url is None):
        print(ACADEMIC_USE_NOTICE)
        print(f"Repository: {OFFICIAL_REPOSITORY}")
        for name, url in OFFICIAL_SOURCES.items():
            print(f"{name}: {url}")
        if args.archive is None and args.url is None:
            print("\nDownload an archive, then rerun with --archive and --accept-academic-use.")
        return 0

    try:
        result = acquire_archive(
            destination=args.destination,
            accept_academic_use=args.accept_academic_use,
            archive=args.archive,
            url=args.url,
            expected_sha256=args.sha256,
            keep_archive=not args.delete_downloaded_archive,
        )
    except AcquisitionError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
