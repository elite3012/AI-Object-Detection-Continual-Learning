from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


def test_class_review_matches_the_locked_shortlist() -> None:
    root = Path(__file__).resolve().parents[2]
    subset = yaml.safe_load((root / "configs/data/ip102_subset.yaml").read_text(encoding="utf-8"))
    review = yaml.safe_load(
        (root / "configs/data/ip102_class_review.yaml").read_text(encoding="utf-8")
    )

    classes = review["classes"]
    reviewed_ids = [item["ip102_id"] for item in classes]
    locked_ids = subset["shortlist"]["manual_class_ids"]

    assert review["status"] == "approved_for_baseline"
    assert reviewed_ids == locked_ids
    assert len(reviewed_ids) == len(set(reviewed_ids)) == subset["shortlist"]["target_classes"]
    assert Counter(item["stratum"] for item in classes) == {
        "head": 4,
        "middle": 4,
        "tail": 4,
    }
    assert all(item["canonical_name"] and item["common_name_vi"] for item in classes)
    assert all(item["external_source"]["license"] for item in classes)
    assert all(item["external_source"]["url"].startswith("https://") for item in classes)
