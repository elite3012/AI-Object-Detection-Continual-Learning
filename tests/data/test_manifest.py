from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from pestscope.data.audit import ShortlistPolicy, analyze_manifest
from pestscope.data.manifest import ManifestRecord, build_manifest, read_manifest


def _pattern(path: Path, seed: int) -> None:
    image = Image.new("RGB", (32, 24), (20 + seed, 40, 60))
    draw = ImageDraw.Draw(image)
    draw.rectangle((seed % 10, 3, 18 + seed % 8, 18), fill=(180, 30 + seed, 80))
    image.save(path)


def test_manifest_supports_zero_based_labels_and_reports_leakage(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (tmp_path / "classes.txt").write_text("1 first pest\n2 second pest\n", encoding="utf-8")

    _pattern(images / "first-train.png", 1)
    (images / "first-val.png").write_bytes((images / "first-train.png").read_bytes())
    _pattern(images / "first-test.png", 2)
    _pattern(images / "second-train.png", 20)
    _pattern(images / "second-val.png", 21)
    _pattern(images / "second-test.png", 22)

    (tmp_path / "train.txt").write_text(
        "images/first-train.png 0\nimages/second-train.png 1\n", encoding="utf-8"
    )
    (tmp_path / "val.txt").write_text(
        "0 images/first-val.png\n1 images/second-val.png\n", encoding="utf-8"
    )
    (tmp_path / "test.txt").write_text(
        "images/first-test.png,0\nimages/second-test.png,1\n", encoding="utf-8"
    )

    manifest_path = tmp_path / "manifest.csv"
    records = build_manifest(
        dataset_root=tmp_path,
        classes_file=Path("classes.txt"),
        images_root=Path("images"),
        split_files={
            "train": Path("train.txt"),
            "val": Path("val.txt"),
            "test": Path("test.txt"),
        },
        output=manifest_path,
        workers=2,
    )
    audit = analyze_manifest(
        records,
        policy=ShortlistPolicy(
            target_classes=2,
            min_train=1,
            min_val=1,
            min_test=1,
            max_exact_cross_split_duplicate_records=0,
        ),
    )

    assert len(read_manifest(manifest_path)) == 6
    assert {record.class_id for record in records} == {1, 2}
    assert audit["exact_cross_split_group_count"] == 1
    first = next(row for row in audit["classes"] if row["class_id"] == 1)
    assert not first["eligible"]
    assert "exact_cross_split_duplicate" in first["ineligible_reasons"]


def test_manifest_records_missing_and_corrupt_images(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (tmp_path / "classes.txt").write_text("1 pest\n", encoding="utf-8")
    (images / "broken.jpg").write_text("not an image", encoding="utf-8")
    for split, filename in {
        "train": "broken.jpg",
        "val": "missing.jpg",
        "test": "missing-too.jpg",
    }.items():
        (tmp_path / f"{split}.txt").write_text(f"images/{filename} 1\n", encoding="utf-8")

    records = build_manifest(
        dataset_root=tmp_path,
        classes_file=Path("classes.txt"),
        images_root=Path("images"),
        split_files={name: Path(f"{name}.txt") for name in ("train", "val", "test")},
        output=tmp_path / "manifest.csv",
        label_base="one",
        workers=1,
    )

    assert {record.status for record in records} == {"corrupt", "missing"}


def _record(class_id: int, split: str, index: int, train_rank: int) -> ManifestRecord:
    split_offset = {"train": 0, "val": 100, "test": 200}[split]
    return ManifestRecord(
        image_id=f"{class_id}-{split}-{index}",
        path=f"{class_id}/{split}-{index}.png",
        split=split,
        class_id=class_id,
        class_name=f"pest {class_id}",
        size_bytes=100 + train_rank,
        width=32,
        height=24,
        sha256=f"{class_id:02x}{split}{index}",
        dhash=f"{class_id * 1000 + split_offset + index:016x}",
        status="ok",
        error=None,
    )


def test_shortlist_represents_head_middle_and_tail_classes() -> None:
    records = []
    for class_id in range(1, 10):
        train_count = 20 - class_id
        records.extend(
            _record(class_id, "train", index, train_count) for index in range(train_count)
        )
        records.extend(_record(class_id, "val", index, train_count) for index in range(3))
        records.extend(_record(class_id, "test", index, train_count) for index in range(3))

    audit = analyze_manifest(
        records,
        policy=ShortlistPolicy(target_classes=6, min_train=5, min_val=2, min_test=2),
        near_duplicate_distance=0,
    )
    selected = [row for row in audit["classes"] if row["provisional_selected"]]

    assert len(selected) == 6
    assert {row["stratum"] for row in selected} == {"head", "middle", "tail"}
