from __future__ import annotations

import csv
import hashlib
import shlex
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MANIFEST_FIELDS = (
    "image_id",
    "path",
    "split",
    "class_id",
    "class_name",
    "size_bytes",
    "width",
    "height",
    "sha256",
    "dhash",
    "status",
    "error",
)


@dataclass(frozen=True)
class ClassDefinition:
    class_id: int
    name: str


@dataclass(frozen=True)
class RawSplitRecord:
    path: str
    split: str
    raw_class_id: int
    source_line: int


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    path: str
    split: str
    class_id: int
    class_name: str
    size_bytes: int | None
    width: int | None
    height: int | None
    sha256: str | None
    dhash: str | None
    status: str
    error: str | None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


class ManifestError(ValueError):
    """Raised when IP102 metadata cannot be interpreted unambiguously."""


def read_classes(path: Path) -> dict[int, ClassDefinition]:
    classes: dict[int, ClassDefinition] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise ManifestError(f"Invalid class definition at {path}:{line_number}: {raw_line!r}")
        class_id = int(parts[0])
        name = " ".join(parts[1].split())
        if class_id in classes:
            raise ManifestError(f"Duplicate class id {class_id} in {path}")
        classes[class_id] = ClassDefinition(class_id=class_id, name=name)

    if not classes:
        raise ManifestError(f"No class definitions found in {path}")
    return classes


def _split_tokens(line: str) -> list[str]:
    if "," in line:
        return [token.strip() for token in next(csv.reader([line])) if token.strip()]
    return shlex.split(line)


def read_split(path: Path, split: str) -> list[RawSplitRecord]:
    records: list[RawSplitRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = _split_tokens(line)
        if len(tokens) != 2:
            raise ManifestError(
                f"Expected image path and label at {path}:{line_number}, got {raw_line!r}"
            )

        first_is_id = tokens[0].lstrip("+-").isdigit()
        last_is_id = tokens[1].lstrip("+-").isdigit()
        if first_is_id == last_is_id:
            raise ManifestError(
                f"Could not identify one label id at {path}:{line_number}: {raw_line!r}"
            )
        if first_is_id:
            raw_class_id = int(tokens[0])
            image_path = tokens[1]
        else:
            image_path = tokens[0]
            raw_class_id = int(tokens[1])

        records.append(
            RawSplitRecord(
                path=image_path.replace("\\", "/").removeprefix("./"),
                split=split,
                raw_class_id=raw_class_id,
                source_line=line_number,
            )
        )

    if not records:
        raise ManifestError(f"No records found in split file {path}")
    return records


def _label_offset(
    raw_records: list[RawSplitRecord],
    classes: dict[int, ClassDefinition],
    label_base: str,
) -> int:
    if label_base not in {"auto", "zero", "one"}:
        raise ManifestError("label_base must be auto, zero, or one")
    if label_base == "zero":
        return 1
    if label_base == "one":
        return 0

    raw_ids = {record.raw_class_id for record in raw_records}
    one_based = raw_ids.issubset(classes)
    zero_based = {value + 1 for value in raw_ids}.issubset(classes)
    if one_based and not zero_based:
        return 0
    if zero_based and not one_based:
        return 1
    if 0 in raw_ids and zero_based:
        return 1
    raise ManifestError(
        "Label base is ambiguous. Pass label_base='zero' or label_base='one' explicitly."
    )


def _resolve_metadata_path(dataset_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def _resolve_image_path(dataset_root: Path, images_root: Path, relative_path: str) -> Path:
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        return dataset_root / "__invalid_path__"

    candidates = (
        dataset_root / candidate_path,
        images_root / candidate_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(image: Image.Image) -> str:
    grayscale = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


def _inspect_record(
    raw: RawSplitRecord,
    class_id: int,
    class_name: str,
    dataset_root: Path,
    images_root: Path,
) -> ManifestRecord:
    absolute_path = _resolve_image_path(dataset_root, images_root, raw.path)
    try:
        stored_path = absolute_path.relative_to(dataset_root).as_posix()
    except ValueError:
        stored_path = raw.path
    image_id = hashlib.sha256(f"{raw.split}:{stored_path}".encode()).hexdigest()[:20]

    if not absolute_path.is_file():
        return ManifestRecord(
            image_id=image_id,
            path=stored_path,
            split=raw.split,
            class_id=class_id,
            class_name=class_name,
            size_bytes=None,
            width=None,
            height=None,
            sha256=None,
            dhash=None,
            status="missing",
            error=f"Image not found for split line {raw.source_line}",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(absolute_path) as image:
                image.load()
                width, height = image.size
                dhash = _difference_hash(image)
        return ManifestRecord(
            image_id=image_id,
            path=stored_path,
            split=raw.split,
            class_id=class_id,
            class_name=class_name,
            size_bytes=absolute_path.stat().st_size,
            width=width,
            height=height,
            sha256=_sha256(absolute_path),
            dhash=dhash,
            status="ok",
            error=None,
        )
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        return ManifestRecord(
            image_id=image_id,
            path=stored_path,
            split=raw.split,
            class_id=class_id,
            class_name=class_name,
            size_bytes=absolute_path.stat().st_size,
            width=None,
            height=None,
            sha256=_sha256(absolute_path),
            dhash=None,
            status="corrupt",
            error=str(exc),
        )


def write_manifest(records: list[ManifestRecord], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)
    temporary.replace(output)
    return output


def build_manifest(
    *,
    dataset_root: Path,
    classes_file: Path,
    images_root: Path,
    split_files: dict[str, Path],
    output: Path,
    label_base: str = "auto",
    workers: int = 4,
) -> list[ManifestRecord]:
    dataset_root = dataset_root.resolve()
    classes_path = _resolve_metadata_path(dataset_root, classes_file)
    images_path = _resolve_metadata_path(dataset_root, images_root)
    classes = read_classes(classes_path)
    raw_records = [
        record
        for split, path in split_files.items()
        for record in read_split(_resolve_metadata_path(dataset_root, path), split)
    ]
    offset = _label_offset(raw_records, classes, label_base)

    normalized: list[tuple[RawSplitRecord, int, str]] = []
    for raw in raw_records:
        class_id = raw.raw_class_id + offset
        definition = classes.get(class_id)
        if definition is None:
            raise ManifestError(
                f"Unknown class id {raw.raw_class_id} in {raw.split} line {raw.source_line}"
            )
        normalized.append((raw, class_id, definition.name))

    if workers < 1:
        raise ManifestError("workers must be at least 1")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(
            executor.map(
                lambda item: _inspect_record(*item, dataset_root, images_path),
                normalized,
            )
        )
    records.sort(key=lambda item: (item.split, item.class_id, item.path))
    write_manifest(records, output)
    return records


def read_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            records.append(
                ManifestRecord(
                    image_id=row["image_id"],
                    path=row["path"],
                    split=row["split"],
                    class_id=int(row["class_id"]),
                    class_name=row["class_name"],
                    size_bytes=int(row["size_bytes"]) if row["size_bytes"] else None,
                    width=int(row["width"]) if row["width"] else None,
                    height=int(row["height"]) if row["height"] else None,
                    sha256=row["sha256"] or None,
                    dhash=row["dhash"] or None,
                    status=row["status"],
                    error=row["error"] or None,
                )
            )
    return records
