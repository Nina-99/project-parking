# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Clean template renderer for the Bolivian national plate.

Draws at high resolution (fixed 610x204, 2x the physical 305x102 ratio) so every
downstream degradation starts from the same sharp master. The caller downsamples
to the target real-world size afterwards — that downsampling IS the pixelation,
not an extra filter.

Approximations, stated so nobody mistakes them for the official die: the font
is a condensed bold stand-in, there is no hologram or reflective sheeting, and
the flag is reduced to a tricolor bar.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Physical ratio 305x102, rendered at 2x for a sharp master.
MASTER_W, MASTER_H = 610, 204
# Where the number sits, read off the real crops by eye and not by threshold:
# the glyphs span ~84% of the width and ~52% of the height, with the header band
# above them. Measuring this with an ink threshold is a trap -- the plate's own
# frame is ink too, so it reports ~94% x 63% and, taken at face value, pushes the
# first and last glyph into the border where the perspective warp amputates them.
GLYPH_WIDTH_FRACTION = 0.84
GLYPH_HEIGHT_FRACTION = 0.52
HEADER_H = 46
# Bolivian particular: blue on white (openvan reports 1852PHD-ish blue).
# Darkened toward what the recorder actually sees: under gate lighting the
# sheeting reads gray and the ink near-black, never vivid studio blue.
BLUE = (11, 47, 110)
WHITE = (212, 212, 212)

# Curated on purpose: every entry is a condensed or plain bold sans that could
# plausibly be a plate die. Exotic faces would teach the model glyph shapes no
# recorder ever sees, which is worse than having fewer fonts.
FONT_CANDIDATES = (
    "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/Verdanab.TTF",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
)


def available_fonts() -> list[str]:
    """The curated faces actually installed, in preference order."""
    return [path for path in FONT_CANDIDATES if Path(path).exists()]


def _load_font(
    size: int, family: str | None = None
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [family] if family else []
    for path in [*candidates, *available_fonts()]:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _centered(
    draw: ImageDraw.ImageDraw,
    cx: float,
    y: float,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    draw.text(
        (cx - (box[2] - box[0]) / 2.0 - box[0], y),
        text,
        font=font,
        fill=fill,
        stroke_width=1,
        stroke_fill=fill,
    )


def _fit_font(
    text: str, family: str | None, target_h: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Font whose rendered cap height lands on ``target_h``.

    Measuring and correcting beats a table of sizes per string length: the
    available faces have different metrics, and the plate number can be six or
    seven characters.
    """
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    size = max(8, int(round(target_h / 0.72)))
    font = _load_font(size, family)
    for _ in range(4):
        box = probe.textbbox((0, 0), text, font=font, stroke_width=1)
        height = box[3] - box[1]
        if height <= 0:
            break
        corrected = max(8, int(round(size * target_h / height)))
        if corrected == size:
            break
        size = corrected
        font = _load_font(size, family)
    return font


def _draw_main_text(img: Image.Image, text: str, rng: random.Random, family: str | None) -> dict:
    """Draw the plate number with die-to-die variation.

    Two invariants come from the real crops, not from taste: the glyph band
    fills ~92% of the width and ~60% of the plate height. An earlier version
    drew the number at 63% x 29%, which after the same downsampling left the
    synthetic glyphs half the size of the real ones -- a domain gap the model
    pays for at every step.

    Character spacing matters for CTC: two plates of the same number printed on
    different dies differ in kerning and weight, and a model that only ever sees
    one spacing learns the spacing instead of the glyphs. So each character is
    placed individually with a jittered gap, and the line is squeezed to fit.
    The result is centred and never leaves the frame.
    """
    target_h = int(MASTER_H * GLYPH_HEIGHT_FRACTION)
    target_w = int(MASTER_W * GLYPH_WIDTH_FRACTION)
    font = _fit_font(text, family, target_h)
    stroke = rng.choice((0, 1, 1, 2))
    gap = rng.uniform(-2.0, 3.0)

    # The layer must be as wide as the line, not as wide as the plate. Sizing it
    # to MASTER_W and then squeezing clips the first and last glyph before there
    # is anything to squeeze -- the glyphs were amputated, not compressed.
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    widths = []
    for char in text:
        box = probe.textbbox((0, 0), char, font=font, stroke_width=stroke)
        widths.append(box[2] - box[0])
    total = sum(widths) + gap * max(0, len(text) - 1)
    pad = 8
    layer = Image.new("RGBA", (max(8, int(total) + 2 * pad), target_h * 3), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x = float(pad)
    for char, width in zip(text, widths, strict=True):
        draw.text(
            (x, pad + target_h // 2),
            char,
            font=font,
            fill=(*BLUE, 255),
            stroke_width=stroke,
            stroke_fill=(*BLUE, 255),
        )
        x += width + gap

    bbox = layer.getbbox()
    if bbox is None:
        return {
            "font": "none",
            "stroke": stroke,
            "gap": round(gap, 3),
            "squeeze": 1.0,
            "text_bbox": [0, 0, 0, 0],
        }
    layer = layer.crop(bbox)
    squeeze = min(1.0, target_w / layer.width)
    if squeeze < 0.999:
        layer = layer.resize((max(1, int(layer.width * squeeze)), layer.height), Image.BICUBIC)

    x0 = int((MASTER_W - layer.width) / 2)
    y0 = HEADER_H + max(0, (MASTER_H - HEADER_H - layer.height) // 2)
    img.alpha_composite(layer, (x0, y0))
    return {
        "font": Path(family).name if family else "default",
        "stroke": stroke,
        "gap": round(gap, 3),
        "squeeze": round(squeeze, 3),
        "text_bbox": [x0, y0, x0 + layer.width, y0 + layer.height],
    }


def render_clean(text: str, dept: str, rng: random.Random | None = None) -> tuple[np.ndarray, dict]:
    """Draw the sharp master (BGR, ``MASTER_W`` x ``MASTER_H``).

    ``rng`` selects the typeface and the die variation. Passing the same rng
    state twice produces the same plate, so a batch can be regenerated or split
    across processes.
    """
    rng = rng or random.Random(0)
    fonts = available_fonts()
    family = rng.choice(fonts) if fonts else None

    img = Image.new("RGBA", (MASTER_W, MASTER_H), (*WHITE, 255))
    draw = ImageDraw.Draw(img)

    # Outer frame.
    draw.rectangle([4, 4, MASTER_W - 5, MASTER_H - 5], outline=BLUE, width=6)

    # Header: tricolor bar + BOLIVIA + department sticker box.
    draw.rectangle([18, 14, 58, 44], fill=(214, 40, 40))
    draw.rectangle([18, 24, 58, 34], fill=(255, 235, 59))
    draw.rectangle([18, 29, 58, 44], fill=(56, 142, 60))
    _centered(draw, MASTER_W / 2.0 - 30, 12, "BOLIVIA", _load_font(30, family), BLUE)
    draw.rectangle([MASTER_W - 78, 12, MASTER_W - 18, 46], outline=BLUE, width=3)
    _centered(draw, MASTER_W - 48, 14, dept, _load_font(26, family), BLUE)

    # Main text, centered in the remaining band. Seven characters get a smaller
    # cut so they never touch the frame: the recorder already clips edges.
    die = _draw_main_text(img, text, rng, family)

    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy(), die
