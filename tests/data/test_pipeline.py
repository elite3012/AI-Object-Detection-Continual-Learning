from __future__ import annotations

from PIL import Image, ImageDraw

from pestscope.data.config import load_data_config
from pestscope.data.pipeline import run_data_pipeline


def test_configured_pipeline_writes_all_review_artifacts(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    (dataset / "classes.txt").write_text("1 pest one\n2 pest two\n3 pest three\n", encoding="utf-8")

    split_lines = {"train": [], "val": [], "test": []}
    for class_id in range(1, 4):
        for split in split_lines:
            filename = f"{split}-{class_id}.png"
            image = Image.new("RGB", (24, 18), (class_id * 60, len(split) * 20, 30))
            draw = ImageDraw.Draw(image)
            split_offset = {"train": 2, "val": 7, "test": 12}[split]
            draw.rectangle(
                (split_offset, class_id, split_offset + 5, class_id + 8),
                fill=(20, 220 - class_id * 20, 90),
            )
            image.save(images / filename)
            split_lines[split].append(f"images/{filename} {class_id}")
    for split, lines in split_lines.items():
        (dataset / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset:
  root: {dataset.as_posix()}
  classes_file: classes.txt
  images_root: images
  split_files:
    train: train.txt
    val: val.txt
    test: test.txt
manifest:
  label_base: one
  workers: 1
audit:
  near_duplicate_distance: 0
  near_duplicate_report_limit: 20
shortlist:
  target_classes: 3
  min_train: 1
  min_val: 1
  min_test: 1
  max_exact_cross_split_duplicate_records: 0
  manual_class_ids: [1, 2, 3]
outputs:
  manifest: {output.as_posix()}/manifest.csv
  audit_json: {output.as_posix()}/audit.json
  audit_markdown: {output.as_posix()}/eda.md
  shortlist: {output.as_posix()}/shortlist.csv
""".strip(),
        encoding="utf-8",
    )

    result = run_data_pipeline(load_data_config(config_path))

    assert result["records"] == 9
    assert result["selected_classes"] == 3
    assert {path.name for path in output.iterdir()} == {
        "manifest.csv",
        "audit.json",
        "eda.md",
        "shortlist.csv",
    }

    reused = run_data_pipeline(load_data_config(config_path), rebuild_manifest=False)
    assert reused["records"] == 9
    assert reused["selected_classes"] == 3
