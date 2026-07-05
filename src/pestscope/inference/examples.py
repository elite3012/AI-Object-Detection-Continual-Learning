from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

BUNDLED_DEMO_DIR = Path(__file__).resolve().parents[3] / "assets" / "demo_examples"
EXAMPLE_PALETTE = {
    1: ((49, 110, 69), (197, 231, 202)),
    8: ((93, 83, 43), (232, 216, 164)),
    40: ((110, 62, 53), (238, 190, 176)),
    77: ((77, 90, 108), (218, 226, 236)),
}


@dataclass(frozen=True)
class DemoExample:
    id: str
    class_id: int
    title: str
    subtitle: str
    expected_name: str
    image_url: str
    provider: str
    license: str
    source_url: str
    image_kind: str
    dataset_image_path: Path | None = None
    attribution: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class_id": self.class_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "expected_name": self.expected_name,
            "image_url": self.image_url,
            "provider": self.provider,
            "license": self.license,
            "source_url": self.source_url,
            "image_kind": self.image_kind,
            "attribution": self.attribution,
        }


def load_demo_examples(
    class_review_path: Path,
    *,
    dataset_root: Path | None = None,
    manifest_path: Path | None = None,
) -> list[DemoExample]:
    raw = yaml.safe_load(class_review_path.read_text(encoding="utf-8"))
    dataset_images = _dataset_images_by_class(dataset_root, manifest_path)
    examples = []
    for item in raw["classes"]:
        class_id = int(item["ip102_id"])
        source = item["external_source"]
        bundled_photo = _bundled_path(class_id).is_file()
        dataset_image_path = dataset_images.get(class_id)
        image_kind = "photo" if bundled_photo else "dataset" if dataset_image_path else "reference"
        provider = "Local IP102 manifest" if image_kind == "dataset" else str(source["provider"])
        license_name = (
            "Local IP102 dataset file" if image_kind == "dataset" else str(source["license"])
        )
        examples.append(
            DemoExample(
                id=f"class-{class_id}",
                class_id=class_id,
                title=str(item["common_name_en"]),
                subtitle=str(item["canonical_name"]),
                expected_name=str(item["common_name_en"]),
                image_url=f"/api/v1/examples/class-{class_id}/image",
                provider=provider,
                license=license_name,
                source_url=str(source["url"]),
                image_kind=image_kind,
                dataset_image_path=dataset_image_path,
                attribution=source.get("attribution"),
            )
        )
    return examples


def _external_image_bytes(example: DemoExample, cache_dir: Path) -> bytes | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{example.id}.jpg"
    if cached.is_file():
        return cached.read_bytes()

    request = Request(
        example.source_url,
        headers={"User-Agent": "PestScope-IP102-demo/0.1"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            raw = response.read(8 * 1024 * 1024)
    except (OSError, URLError, TimeoutError):
        return None

    try:
        with Image.open(BytesIO(raw)) as image:
            clean = ImageOps.exif_transpose(image).convert("RGB")
            clean.thumbnail((900, 650), Image.Resampling.LANCZOS)
            output = BytesIO()
            clean.save(output, format="JPEG", quality=86, optimize=True)
    except OSError:
        return None
    cached.write_bytes(output.getvalue())
    return output.getvalue()


def _bundled_image_bytes(example: DemoExample) -> bytes | None:
    bundled = _bundled_path(example.class_id)
    if bundled.is_file():
        return bundled.read_bytes()
    return None


def _dataset_images_by_class(
    dataset_root: Path | None,
    manifest_path: Path | None,
) -> dict[int, Path]:
    if dataset_root is None or manifest_path is None:
        return {}
    if not dataset_root.is_dir() or not manifest_path.is_file():
        return {}

    selected: dict[int, tuple[int, Path]] = {}
    split_rank = {"test": 0, "val": 1, "train": 2}
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue
            try:
                class_id = int(str(row["class_id"]).strip())
            except (KeyError, TypeError, ValueError):
                continue
            relative_path = row.get("path") or row.get("image_path") or row.get("relative_path")
            if not relative_path:
                continue
            image_path = dataset_root / relative_path
            if not image_path.is_file():
                continue
            rank = split_rank.get(str(row.get("split", "")).lower(), 9)
            current = selected.get(class_id)
            if current is None or rank < current[0]:
                selected[class_id] = (rank, image_path)
    return {class_id: path for class_id, (_, path) in selected.items()}


def _dataset_image_bytes(example: DemoExample) -> bytes | None:
    if example.dataset_image_path is None or not example.dataset_image_path.is_file():
        return None
    try:
        with Image.open(example.dataset_image_path) as image:
            clean = ImageOps.exif_transpose(image).convert("RGB")
            canvas = _fit_dataset_preview(clean)
            output = BytesIO()
            canvas.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue()
    except OSError:
        return None


def _fit_dataset_preview(image: Image.Image, size: tuple[int, int] = (900, 650)) -> Image.Image:
    background = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(radius=18))
    wash = Image.new("RGB", size, (245, 250, 241))
    canvas = Image.blend(background, wash, 0.34)

    scale = min(size[0] / image.width, size[1] / image.height)
    preview_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    preview = image.resize(preview_size, Image.Resampling.LANCZOS)
    x = (size[0] - preview.width) // 2
    y = (size[1] - preview.height) // 2
    canvas.paste(preview, (x, y))
    return canvas


def _bundled_path(class_id: int) -> Path:
    return BUNDLED_DEMO_DIR / f"class-{class_id}.jpg"


def _fallback_image(example: DemoExample) -> bytes:
    dark, light = EXAMPLE_PALETTE.get(example.class_id, _palette_from_id(example.class_id))
    image = Image.new("RGB", (900, 650), (245, 250, 241))
    draw = ImageDraw.Draw(image)
    _draw_field_plate(draw, light)
    _draw_specimen(draw, example.class_id, dark)
    _draw_reference_label(draw, example)
    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def _palette_from_id(class_id: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    digest = hashlib.sha256(str(class_id).encode("ascii")).digest()
    dark = (45 + digest[0] % 70, 60 + digest[1] % 70, 55 + digest[2] % 70)
    light = (218 + digest[3] % 25, 228 + digest[4] % 22, 218 + digest[5] % 25)
    return dark, light


def _draw_field_plate(draw: ImageDraw.ImageDraw, light: tuple[int, int, int]) -> None:
    draw.rectangle((0, 0, 900, 650), fill=light)
    for index in range(-7, 24):
        x = index * 58
        draw.line(
            (x, 0, x + 360, 650),
            fill=tuple(max(0, value - 18) for value in light),
            width=3,
        )
    for index in range(8):
        y = 72 + index * 64
        draw.arc(
            (110, y - 90, 790, y + 260),
            start=205,
            end=335,
            fill=tuple(max(0, value - 26) for value in light),
            width=2,
        )
    draw.rectangle((22, 22, 878, 628), outline=(211, 224, 211), width=3)
    draw.rectangle((42, 42, 858, 608), outline=(238, 245, 235), width=2)


def _draw_specimen(
    draw: ImageDraw.ImageDraw,
    class_id: int,
    dark: tuple[int, int, int],
) -> None:
    accent = tuple(min(255, value + 56) for value in dark)
    shadow = tuple(max(0, value - 24) for value in dark)
    center_x = 450
    if class_id in {1, 40, 87, 89, 98}:
        draw.polygon([(455, 255), (174, 164), (310, 414)], fill=accent, outline=shadow)
        draw.polygon([(445, 255), (726, 164), (590, 414)], fill=accent, outline=shadow)
        for offset in (-138, 138):
            draw.line((center_x, 285, center_x + offset, 218), fill=shadow, width=5)
            draw.line((center_x, 326, center_x + offset, 378), fill=shadow, width=5)
        draw.ellipse((372, 178, 528, 426), fill=dark, outline=shadow, width=4)
        draw.ellipse((414, 212, 486, 308), fill=accent)
    elif class_id in {8, 58, 72, 83, 85}:
        draw.ellipse((318, 210, 582, 410), fill=dark, outline=shadow, width=4)
        draw.ellipse((385, 145, 515, 275), fill=accent, outline=shadow, width=4)
        for side in (-1, 1):
            for row, y in enumerate((246, 296, 346)):
                draw.line(
                    (center_x + side * 96, y, center_x + side * (196 + row * 22), y - 48),
                    fill=shadow,
                    width=6,
                )
                draw.line(
                    (
                        center_x + side * 96,
                        y + 18,
                        center_x + side * (206 + row * 18),
                        y + 84,
                    ),
                    fill=shadow,
                    width=6,
                )
        draw.line((410, 182, 338, 112), fill=shadow, width=4)
        draw.line((490, 182, 562, 112), fill=shadow, width=4)
    else:
        draw.ellipse((290, 188, 610, 438), fill=(238, 242, 227), outline=shadow, width=5)
        draw.ellipse((340, 232, 560, 388), fill=dark)
        for x in range(350, 560, 38):
            draw.line((x, 236, x - 28, 386), fill=accent, width=2)
        for side in (-1, 1):
            draw.line(
                (center_x + side * 112, 288, center_x + side * 238, 236),
                fill=shadow,
                width=7,
            )
            draw.line(
                (center_x + side * 114, 340, center_x + side * 226, 426),
                fill=shadow,
                width=7,
            )


def _draw_reference_label(draw: ImageDraw.ImageDraw, example: DemoExample) -> None:
    heading_font, body_font, mono_font = _fonts()
    draw.rounded_rectangle(
        (44, 470, 856, 596),
        radius=12,
        fill=(252, 254, 248),
        outline=(211, 224, 211),
        width=2,
    )
    draw.rounded_rectangle(
        (64, 492, 216, 526),
        radius=8,
        fill=(230, 241, 225),
        outline=(191, 216, 195),
        width=1,
    )
    draw.text((82, 501), "REFERENCE", fill=(39, 91, 60), font=mono_font)
    draw.text((244, 490), example.expected_name[:44], fill=(23, 36, 31), font=heading_font)
    draw.text((244, 528), example.subtitle[:68], fill=(92, 111, 101), font=body_font)
    draw.text((70, 552), f"IP102 class {example.class_id}", fill=(92, 111, 101), font=body_font)


def _fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    body_candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    try:
        heading = ImageFont.truetype(str(next(path for path in candidates if path.is_file())), 28)
        body = ImageFont.truetype(str(next(path for path in body_candidates if path.is_file())), 20)
        mono = ImageFont.truetype(str(next(path for path in candidates if path.is_file())), 14)
        return heading, body, mono
    except (OSError, StopIteration):
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback


def example_image_bytes(
    example: DemoExample,
    *,
    cache_dir: Path,
    fetch_external: bool,
) -> bytes:
    bundled = _bundled_image_bytes(example)
    if bundled is not None:
        return bundled
    dataset = _dataset_image_bytes(example)
    if dataset is not None:
        return dataset
    if fetch_external:
        external = _external_image_bytes(example, cache_dir)
        if external is not None:
            return external
    return _fallback_image(example)
