from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .manifest import ManifestRecord


@dataclass(frozen=True)
class ShortlistPolicy:
    target_classes: int = 12
    min_train: int = 100
    min_val: int = 20
    min_test: int = 20
    max_exact_cross_split_duplicate_records: int = 0
    manual_class_ids: tuple[int, ...] = ()


class _BKNode:
    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indexes = [index]
        self.children: dict[int, _BKNode] = {}

    def add(self, value: int, index: int) -> None:
        distance = (self.value ^ value).bit_count()
        if distance == 0:
            self.indexes.append(index)
            return
        child = self.children.get(distance)
        if child is None:
            self.children[distance] = _BKNode(value, index)
        else:
            child.add(value, index)

    def query(self, value: int, threshold: int) -> list[int]:
        distance = (self.value ^ value).bit_count()
        matches = list(self.indexes) if distance <= threshold else []
        lower = max(1, distance - threshold)
        upper = distance + threshold
        for edge, child in self.children.items():
            if lower <= edge <= upper:
                matches.extend(child.query(value, threshold))
        return matches


def _exact_duplicate_groups(records: list[ManifestRecord]) -> list[dict]:
    by_hash: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.status == "ok" and record.sha256:
            by_hash[record.sha256].append(record)

    groups = []
    for digest, members in by_hash.items():
        splits = sorted({member.split for member in members})
        if len(splits) < 2:
            continue
        groups.append(
            {
                "sha256": digest,
                "splits": splits,
                "records": [
                    {
                        "image_id": member.image_id,
                        "path": member.path,
                        "split": member.split,
                        "class_id": member.class_id,
                    }
                    for member in members
                ],
            }
        )
    return sorted(groups, key=lambda item: item["sha256"])


def _near_duplicate_pairs(
    records: list[ManifestRecord], threshold: int, report_limit: int
) -> tuple[int, list[dict], Counter[int]]:
    valid = [record for record in records if record.status == "ok" and record.dhash]
    if not valid:
        return 0, [], Counter()

    tree: _BKNode | None = None
    total = 0
    reported: list[dict] = []
    records_by_class: Counter[int] = Counter()
    for index, record in enumerate(valid):
        value = int(record.dhash, 16)
        matches = tree.query(value, threshold) if tree else []
        for previous_index in matches:
            previous = valid[previous_index]
            if previous.split == record.split or previous.sha256 == record.sha256:
                continue
            distance = (int(previous.dhash, 16) ^ value).bit_count()
            total += 1
            records_by_class[previous.class_id] += 1
            records_by_class[record.class_id] += 1
            if len(reported) < report_limit:
                reported.append(
                    {
                        "left_image_id": previous.image_id,
                        "left_path": previous.path,
                        "left_split": previous.split,
                        "left_class_id": previous.class_id,
                        "right_image_id": record.image_id,
                        "right_path": record.path,
                        "right_split": record.split,
                        "right_class_id": record.class_id,
                        "hamming_distance": distance,
                    }
                )
        if tree is None:
            tree = _BKNode(value, index)
        else:
            tree.add(value, index)
    return total, reported, records_by_class


def _stratify(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda item: (-item["train"], item["class_id"]))
    count = len(ordered)
    names = ("head", "middle", "tail")
    for rank, row in enumerate(ordered):
        stratum_index = min(2, (rank * 3) // max(1, count))
        row["stratum"] = names[stratum_index]


def _select_classes(rows: list[dict], policy: ShortlistPolicy) -> None:
    eligible = [row for row in rows if row["eligible"]]
    _stratify(eligible)

    if policy.manual_class_ids:
        if len(policy.manual_class_ids) != len(set(policy.manual_class_ids)):
            raise ValueError("manual_class_ids must not contain duplicates")
        if len(policy.manual_class_ids) != policy.target_classes:
            raise ValueError("manual_class_ids must contain target_classes entries")

        rows_by_id = {row["class_id"]: row for row in rows}
        missing = sorted(set(policy.manual_class_ids) - set(rows_by_id))
        if missing:
            raise ValueError(f"manual_class_ids are absent from the manifest: {missing}")
        blocked = {
            class_id: rows_by_id[class_id]["ineligible_reasons"]
            for class_id in policy.manual_class_ids
            if not rows_by_id[class_id]["eligible"]
        }
        if blocked:
            raise ValueError(f"manual_class_ids include ineligible classes: {blocked}")

        selected_ids = set(policy.manual_class_ids)
        for row in rows:
            row["provisional_selected"] = row["class_id"] in selected_ids
            row["external_validation"] = (
                "reviewed" if row["provisional_selected"] else "not_started"
            )
        return

    quotas = {
        "head": policy.target_classes // 3,
        "middle": policy.target_classes // 3,
        "tail": policy.target_classes // 3,
    }
    for name in ("head", "middle", "tail")[: policy.target_classes % 3]:
        quotas[name] += 1

    selected: list[dict] = []
    for stratum in ("head", "middle", "tail"):
        candidates = sorted(
            (row for row in eligible if row["stratum"] == stratum),
            key=lambda item: (-item["selection_score"], item["class_id"]),
        )
        selected.extend(candidates[: quotas[stratum]])

    if len(selected) < policy.target_classes:
        selected_ids = {row["class_id"] for row in selected}
        remaining = sorted(
            (row for row in eligible if row["class_id"] not in selected_ids),
            key=lambda item: (-item["selection_score"], item["class_id"]),
        )
        selected.extend(remaining[: policy.target_classes - len(selected)])

    selected_ids = {row["class_id"] for row in selected}
    for row in rows:
        row["provisional_selected"] = row["class_id"] in selected_ids
        row["external_validation"] = "pending" if row["provisional_selected"] else "not_started"


def analyze_manifest(
    records: list[ManifestRecord],
    *,
    policy: ShortlistPolicy | None = None,
    near_duplicate_distance: int = 4,
    near_duplicate_report_limit: int = 500,
) -> dict:
    policy = policy or ShortlistPolicy()
    if policy.target_classes < 1:
        raise ValueError("target_classes must be at least 1")
    if not 0 <= near_duplicate_distance <= 64:
        raise ValueError("near_duplicate_distance must be between 0 and 64")

    status_counts = Counter(record.status for record in records)
    split_counts = Counter(record.split for record in records)
    exact_groups = _exact_duplicate_groups(records)
    near_total, near_pairs, near_records_by_class = _near_duplicate_pairs(
        records, near_duplicate_distance, near_duplicate_report_limit
    )

    exact_records_by_class: Counter[int] = Counter()
    for group in exact_groups:
        exact_records_by_class.update(record["class_id"] for record in group["records"])

    counts: dict[int, Counter[str]] = defaultdict(Counter)
    names: dict[int, str] = {}
    corrupt_by_class: Counter[int] = Counter()
    for record in records:
        names[record.class_id] = record.class_name
        if record.status == "ok":
            counts[record.class_id][record.split] += 1
        else:
            corrupt_by_class[record.class_id] += 1

    class_rows = []
    for class_id in sorted(names):
        train = counts[class_id]["train"]
        val = counts[class_id]["val"]
        test = counts[class_id]["test"]
        exact_duplicate_records = exact_records_by_class[class_id]
        near_duplicate_pairs = near_records_by_class[class_id]
        reasons = []
        if train < policy.min_train:
            reasons.append(f"train<{policy.min_train}")
        if val < policy.min_val:
            reasons.append(f"val<{policy.min_val}")
        if test < policy.min_test:
            reasons.append(f"test<{policy.min_test}")
        if corrupt_by_class[class_id]:
            reasons.append("corrupt_or_missing")
        if exact_duplicate_records > policy.max_exact_cross_split_duplicate_records:
            reasons.append("exact_cross_split_duplicate")
        score = (
            math.log1p(train)
            + 0.75 * math.log1p(val)
            + 0.75 * math.log1p(test)
            - 0.5 * exact_duplicate_records
            - 0.05 * near_duplicate_pairs
        )
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "train": train,
                "val": val,
                "test": test,
                "invalid": corrupt_by_class[class_id],
                "exact_cross_split_duplicate_records": exact_duplicate_records,
                "near_cross_split_duplicate_pairs": near_duplicate_pairs,
                "eligible": not reasons,
                "ineligible_reasons": reasons,
                "selection_score": round(score, 6),
                "stratum": None,
            }
        )
    _select_classes(class_rows, policy)

    train_counts = [row["train"] for row in class_rows if row["train"] > 0]
    imbalance_ratio = max(train_counts) / min(train_counts) if train_counts else None
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "class_count": len(class_rows),
        "train_imbalance_ratio": imbalance_ratio,
        "exact_cross_split_group_count": len(exact_groups),
        "exact_cross_split_groups": exact_groups,
        "near_duplicate_distance": near_duplicate_distance,
        "near_cross_split_pair_count": near_total,
        "near_cross_split_pairs_reported": near_pairs,
        "near_cross_split_pairs_truncated": near_total > len(near_pairs),
        "shortlist_policy": {
            "target_classes": policy.target_classes,
            "min_train": policy.min_train,
            "min_val": policy.min_val,
            "min_test": policy.min_test,
            "max_exact_cross_split_duplicate_records": (
                policy.max_exact_cross_split_duplicate_records
            ),
            "manual_class_ids": list(policy.manual_class_ids),
            "selection_mode": "manual" if policy.manual_class_ids else "automatic",
        },
        "classes": class_rows,
    }


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_audit_json(audit: dict, output: Path, manifest_path: Path | None = None) -> Path:
    payload = dict(audit)
    if manifest_path:
        payload["manifest_sha256"] = _manifest_hash(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    return output


def write_audit_markdown(audit: dict, output: Path) -> Path:
    selected = [row for row in audit["classes"] if row["provisional_selected"]]
    manually_reviewed = audit["shortlist_policy"].get("selection_mode") == "manual"
    shortlist_heading = "Reviewed shortlist" if manually_reviewed else "Provisional shortlist"
    shortlist_note = (
        "This set was locked after count, leakage, taxonomy, and external-source review."
        if manually_reviewed
        else (
            "This list deliberately samples eligible head, middle, and tail classes. It is not "
            "final until taxonomy and external-image availability are reviewed."
        )
    )
    lines = [
        "# IP102 Data Audit",
        "",
        f"Generated: `{audit['created_at']}`",
        "",
        "## Dataset health",
        "",
        f"- Records: **{audit['records']}**",
        f"- Classes observed: **{audit['class_count']}**",
        f"- Split counts: `{json.dumps(audit['split_counts'], sort_keys=True)}`",
        f"- Record status: `{json.dumps(audit['status_counts'], sort_keys=True)}`",
        f"- Train imbalance ratio: **{audit['train_imbalance_ratio'] or 'n/a'}**",
        f"- Exact cross-split duplicate groups: **{audit['exact_cross_split_group_count']}**",
        f"- Near cross-split duplicate pairs: **{audit['near_cross_split_pair_count']}**",
        "",
        f"## {shortlist_heading}",
        "",
        shortlist_note,
        "",
        "| ID | Class | Stratum | Train | Val | Test | External set |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['class_id']} | {row['class_name']} | {row['stratum']} | "
        f"{row['train']} | {row['val']} | {row['test']} | {row['external_validation']} |"
        for row in selected
    )
    if manually_reviewed:
        lines.extend(
            [
                "",
                "## Controls carried forward",
                "",
                "- Canonical display names and licensed sources live in the class review config.",
                "- Duplicate-blocked classes remain excluded; source split files stay unchanged.",
                "- Keep the official test split closed until model and thresholds are frozen.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Approval blockers",
                "",
                "- Review every selected class for taxonomic ambiguity and Vietnamese naming.",
                "- Confirm at least one correctly licensed external source per selected class.",
                "- Resolve every exact cross-split duplicate before training.",
                "- Keep the official test split closed until model and thresholds are frozen.",
                "",
                "The generated shortlist is evidence for review, not an automatic declaration "
                "that the classes are suitable for a public field demo.",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_shortlist_csv(audit: dict, output: Path) -> Path:
    fields = (
        "class_id",
        "class_name",
        "stratum",
        "train",
        "val",
        "test",
        "exact_cross_split_duplicate_records",
        "near_cross_split_duplicate_pairs",
        "selection_score",
        "external_validation",
    )
    selected = [row for row in audit["classes"] if row["provisional_selected"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    return output
