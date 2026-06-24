from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from pestscope.data.manifest import ManifestRecord, read_manifest

from .transforms import ImageTransform


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        records: list[ManifestRecord],
        *,
        dataset_root: Path,
        class_to_index: dict[int, int],
        transform: ImageTransform,
    ) -> None:
        self.records = [record for record in records if record.status == "ok"]
        self.dataset_root = dataset_root
        self.class_to_index = class_to_index
        self.transform = transform
        if not self.records:
            raise ValueError("ManifestImageDataset received no valid records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(self.dataset_root / record.path) as image:
            tensor = self.transform(image)
        return tensor, self.class_to_index[record.class_id]


def class_index(selected_class_ids: tuple[int, ...]) -> dict[int, int]:
    return {class_id: index for index, class_id in enumerate(selected_class_ids)}


def selected_records(
    manifest_path: Path,
    *,
    split: str,
    selected_class_ids: tuple[int, ...],
    limit_per_class: int | None = None,
) -> list[ManifestRecord]:
    selected = set(selected_class_ids)
    records = [
        record
        for record in read_manifest(manifest_path)
        if record.split == split and record.class_id in selected and record.status == "ok"
    ]
    if limit_per_class is None:
        return records
    if limit_per_class < 1:
        raise ValueError("limit_per_class must be positive")

    counts: Counter[int] = Counter()
    limited: list[ManifestRecord] = []
    for record in records:
        if counts[record.class_id] >= limit_per_class:
            continue
        counts[record.class_id] += 1
        limited.append(record)
    return limited


def class_counts(records: list[ManifestRecord], selected_class_ids: tuple[int, ...]) -> list[int]:
    counts = Counter(record.class_id for record in records)
    return [counts[class_id] for class_id in selected_class_ids]
