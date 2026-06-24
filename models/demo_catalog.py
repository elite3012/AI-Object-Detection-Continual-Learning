from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DemoSample:
    id: str
    label: str | None
    title: str
    description: str
    expected: str

    def to_dict(self) -> dict:
        return asdict(self)


DEMO_SAMPLES = (
    DemoSample(
        id="connector-pass",
        label="connector-pass",
        title="Connector pass",
        description="Housing and all contact pins are aligned.",
        expected="Known class",
    ),
    DemoSample(
        id="bent-pin",
        label="bent-pin",
        title="Bent contact pin",
        description="One contact is displaced from the pin row.",
        expected="Known class",
    ),
    DemoSample(
        id="burnt-housing",
        label="burnt-housing",
        title="Burnt housing",
        description="Thermal damage is visible around the socket.",
        expected="Known class",
    ),
    DemoSample(
        id="unknown-fastener",
        label=None,
        title="Unknown fastener",
        description="An out-of-scope object used to test rejection.",
        expected="Unknown",
    ),
)

SAMPLE_BY_ID = {sample.id: sample for sample in DEMO_SAMPLES}


def known_demo_samples() -> tuple[DemoSample, ...]:
    return tuple(sample for sample in DEMO_SAMPLES if sample.label is not None)


def render_demo_image(sample_id: str, variant: int = 2) -> Image.Image:
    """Render a repeatable inspection fixture without external image downloads."""
    if sample_id not in SAMPLE_BY_ID:
        raise KeyError(sample_id)

    image = Image.new("RGB", (720, 480), (27, 32, 39))
    draw = ImageDraw.Draw(image)
    shift = (variant - 1) * 7

    for x in range(0, 721, 48):
        draw.line((x, 0, x, 480), fill=(35, 42, 50), width=1)
    for y in range(0, 481, 48):
        draw.line((0, y, 720, y), fill=(35, 42, 50), width=1)

    if sample_id == "unknown-fastener":
        _draw_fastener(draw, shift)
    else:
        _draw_connector(draw, sample_id, shift)

    return image


def demo_image_bytes(sample_id: str, variant: int = 2) -> bytes:
    output = BytesIO()
    render_demo_image(sample_id, variant).save(output, format="PNG", optimize=True)
    return output.getvalue()


def training_images(sample_id: str) -> list[Image.Image]:
    return [render_demo_image(sample_id, variant) for variant in range(3)]


def _draw_connector(draw: ImageDraw.ImageDraw, sample_id: str, shift: int) -> None:
    board = (85 + shift, 120, 635 + shift, 385)
    draw.rounded_rectangle(board, radius=18, fill=(28, 98, 76), outline=(58, 159, 122), width=4)
    for x in range(board[0] + 30, board[2], 62):
        draw.ellipse((x, 345, x + 12, 357), fill=(182, 202, 194))

    housing = (180 + shift, 160, 540 + shift, 330)
    draw.rounded_rectangle(
        housing, radius=16, fill=(201, 207, 211), outline=(242, 244, 245), width=4
    )
    draw.rounded_rectangle(
        (215 + shift, 195, 505 + shift, 292),
        radius=9,
        fill=(64, 72, 81),
        outline=(115, 126, 136),
        width=3,
    )

    pin_x = 242 + shift
    for index in range(6):
        x = pin_x + index * 48
        if sample_id == "bent-pin" and index == 3:
            draw.line((x, 216, x + 17, 268), fill=(225, 180, 75), width=12)
            draw.ellipse((x + 9, 260, x + 25, 276), fill=(225, 180, 75))
        else:
            draw.rounded_rectangle((x, 214, x + 14, 274), radius=4, fill=(225, 180, 75))

    if sample_id == "burnt-housing":
        draw.ellipse((410 + shift, 150, 550 + shift, 288), fill=(68, 43, 31))
        draw.ellipse((438 + shift, 172, 525 + shift, 260), fill=(35, 29, 27))
        draw.line((455 + shift, 164, 490 + shift, 230), fill=(226, 107, 54), width=5)
        draw.line((490 + shift, 230, 530 + shift, 285), fill=(226, 107, 54), width=4)

    draw.rectangle((115 + shift, 98, 265 + shift, 116), fill=(74, 87, 98))
    draw.rectangle((465 + shift, 398, 602 + shift, 410), fill=(74, 87, 98))


def _draw_fastener(draw: ImageDraw.ImageDraw, shift: int) -> None:
    draw.rounded_rectangle(
        (110, 95, 610, 395), radius=22, fill=(43, 59, 76), outline=(74, 96, 117), width=4
    )
    points = [
        (280 + shift, 166),
        (394 + shift, 166),
        (451 + shift, 264),
        (394 + shift, 362),
        (280 + shift, 362),
        (223 + shift, 264),
    ]
    draw.polygon(points, fill=(177, 184, 190), outline=(232, 235, 238))
    draw.ellipse(
        (276 + shift, 203, 398 + shift, 325), fill=(77, 84, 91), outline=(215, 219, 222), width=5
    )
    draw.line((305 + shift, 235, 370 + shift, 292), fill=(210, 215, 219), width=14)
    draw.line((370 + shift, 235, 305 + shift, 292), fill=(210, 215, 219), width=14)
