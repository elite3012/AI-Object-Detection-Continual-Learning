from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

EXAMPLE_CLASS_IDS = (1, 8, 40, 77)
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
            "attribution": self.attribution,
        }


def load_demo_examples(class_review_path: Path) -> list[DemoExample]:
    raw = yaml.safe_load(class_review_path.read_text(encoding="utf-8"))
    rows = {int(item["ip102_id"]): item for item in raw["classes"]}
    examples = []
    for class_id in EXAMPLE_CLASS_IDS:
        item = rows[class_id]
        source = item["external_source"]
        examples.append(
            DemoExample(
                id=f"class-{class_id}",
                class_id=class_id,
                title=str(item["common_name_vi"]),
                subtitle=str(item["canonical_name"]),
                expected_name=str(item["common_name_en"]),
                image_url=f"/api/v1/examples/class-{class_id}/image",
                provider=str(source["provider"]),
                license=str(source["license"]),
                source_url=str(source["url"]),
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


def _fallback_image(example: DemoExample) -> bytes:
    dark, light = EXAMPLE_PALETTE.get(example.class_id, ((40, 56, 70), (230, 235, 239)))
    image = Image.new("RGB", (900, 650), light)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 900, 650), fill=light)
    for index in range(18):
        x = 42 + index * 51
        draw.line((x, 0, x - 160, 650), fill=tuple(max(0, value - 18) for value in light), width=3)
    draw.ellipse((275, 145, 625, 495), fill=dark)
    draw.ellipse((350, 220, 550, 420), fill=tuple(min(255, value + 55) for value in dark))
    draw.line((260, 320, 150, 260), fill=dark, width=9)
    draw.line((640, 320, 760, 260), fill=dark, width=9)
    draw.line((300, 440, 190, 550), fill=dark, width=8)
    draw.line((600, 440, 710, 550), fill=dark, width=8)
    draw.rectangle((28, 28, 430, 126), fill=(255, 255, 255))
    font = ImageFont.load_default()
    draw.text((48, 48), example.expected_name, fill=(20, 28, 36), font=font)
    draw.text((48, 78), example.subtitle, fill=(85, 96, 108), font=font)
    output = BytesIO()
    image.save(output, format="JPEG", quality=88)
    return output.getvalue()


def example_image_bytes(
    example: DemoExample,
    *,
    cache_dir: Path,
    fetch_external: bool,
) -> bytes:
    if fetch_external:
        external = _external_image_bytes(example, cache_dir)
        if external is not None:
            return external
    return _fallback_image(example)
