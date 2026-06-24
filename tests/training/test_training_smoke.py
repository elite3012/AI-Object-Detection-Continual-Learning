from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from pestscope.data.manifest import build_manifest
from pestscope.training import TrainingOverrides, load_training_config, run_training
from pestscope.training.bundle import load_model_bundle


def _make_image(path: Path, color: tuple[int, int, int], marker: int) -> None:
    image = Image.new("RGB", (48, 40), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((marker, marker, marker + 12, marker + 9), fill=(255, 255, 255))
    draw.line((0, 39 - marker, 47, marker), fill=(10, 10, 10), width=2)
    image.save(path)


def _write_review(path: Path) -> None:
    path.write_text(
        """
schema_version: 1
status: fixture
classes:
  - ip102_id: 1
    dataset_label: pest one
    canonical_name: Testus one
    common_name_en: fixture pest one
    common_name_vi: Sau thu nghiem mot
    stratum: head
  - ip102_id: 2
    dataset_label: pest two
    canonical_name: Testus two
    common_name_en: fixture pest two
    common_name_vi: Sau thu nghiem hai
    stratum: middle
  - ip102_id: 3
    dataset_label: pest three
    canonical_name: Testus three
    common_name_en: fixture pest three
    common_name_vi: Sau thu nghiem ba
    stratum: tail
""".strip(),
        encoding="utf-8",
    )


def _write_dataset(root: Path) -> Path:
    images = root / "images"
    images.mkdir(parents=True)
    (root / "classes.txt").write_text(
        "1 pest one\n2 pest two\n3 pest three\n",
        encoding="utf-8",
    )
    split_lines = {"train": [], "val": [], "test": []}
    colors = {1: (220, 40, 50), 2: (40, 190, 80), 3: (50, 80, 220)}
    counts = {"train": 3, "val": 2, "test": 1}
    for split, count in counts.items():
        for class_id, color in colors.items():
            for index in range(count):
                filename = f"{split}-{class_id}-{index}.png"
                _make_image(images / filename, color, marker=class_id * 3 + index)
                split_lines[split].append(f"images/{filename} {class_id}")
    for split, lines in split_lines.items():
        (root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = root.parent / "manifest.csv"
    build_manifest(
        dataset_root=root,
        classes_file=Path("classes.txt"),
        images_root=Path("images"),
        split_files={
            "train": Path("train.txt"),
            "val": Path("val.txt"),
            "test": Path("test.txt"),
        },
        output=manifest_path,
        label_base="one",
        workers=1,
    )
    return manifest_path


def test_training_exports_reloadable_model_bundle(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_path = _write_dataset(dataset_root)
    review_path = tmp_path / "class_review.yaml"
    _write_review(review_path)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
data:
  dataset_root: {dataset_root.as_posix()}
  manifest: {manifest_path.as_posix()}
  class_review: {review_path.as_posix()}
  selected_class_ids: [1, 2, 3]
  image_size: 32
model:
  name: pestnet_s
  width: 8
  dropout: 0.0
training:
  seed: 7
  epochs: 1
  batch_size: 3
  learning_rate: 0.001
  weight_decay: 0.0
  device: cpu
  num_workers: 0
  class_strategy: weighted_loss
outputs:
  run_dir: {(tmp_path / "runs").as_posix()}
  bundle_dir: {(tmp_path / "bundle").as_posix()}
""".strip(),
        encoding="utf-8",
    )

    result = run_training(
        load_training_config(config_path),
        overrides=TrainingOverrides(max_epochs=1),
    )

    bundle_dir = Path(result["bundle"]["bundle_dir"])
    assert (bundle_dir / "model.pt").is_file()
    assert (bundle_dir / "metadata.json").is_file()
    assert (bundle_dir / "metrics.json").is_file()
    assert result["best_validation"]["samples"] == 6
    assert result["parameter_count"] > 0

    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model"]["name"] == "pestnet_s"
    assert [item["ip102_id"] for item in metadata["classes"]] == [1, 2, 3]

    model, loaded_metadata = load_model_bundle(bundle_dir)
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 32, 32))

    assert logits.shape == (1, 3)
    assert loaded_metadata["artifact"]["model_sha256"] == result["bundle"]["model_sha256"]
