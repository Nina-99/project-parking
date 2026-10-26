# SPDX-FileCopyrightText: 2026 Nina
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
# Bolivian particular: blue on white (openvan reports 1852PHD-ish blue).
# Darkened toward what the recorder actually sees: under gate lighting the
# sheeting reads gray and the ink near-black, never vivid studio blue.
BLUE = (11, 47, 110)
WHITE = (235, 235, 235)

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


def _load_font(size: int, family: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [family] if family else []
    for path in [*candidates, *available_fonts()]:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _centered(draw: ImageDraw.ImageDraw, cx: float, y: float, text: str,
              font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
              fill: tuple[int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    draw.text((cx - (box[2] - box[0]) / 2.0 - box[0], y), text, font=font, fill=fill,
              stroke_width=1, stroke_fill=fill)


def _draw_main_text(img: Image.Image, text: str, rng: random.Random,
                    family: str | None) -> dict:
    """Returns the die parameters used, so the manifest can record them."""
    """Draw the plate number with die-to-die variation.

    Character spacing matters for CTC: two plates of the same number printed on
    different dies differ in kerning and weight, and a model that only ever sees
    one spacing learns the spacing instead of the glyphs. So each character is
    placed individually with a jittered gap, and the whole line is stretched
    slightly. The result is still centred and never leaves the frame.
    """
    size = 96 if len(text) <= 6 else 82
    font = _load_font(size, family)
    stroke = rng.choice((0, 1, 1, 2))
    gap = rng.uniform(-2.0, 3.0)
    squeeze = rng.uniform(0.93, 1.07)

    layer = Image.new("RGBA", (MASTER_W, 130), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    widths = []
    for char in text:
        box = draw.textbbox((0, 0), char, font=font, stroke_width=stroke)
        widths.append(box[2] - box[0])
    total = sum(widths) + gap * max(0, len(text) - 1)
    x = (MASTER_W - total) / 2.0
    for char, width in zip(text, widths, strict=True):
        draw.text((x, 0), char, font=font, fill=(*BLUE, 255), stroke_width=stroke,
                  stroke_fill=(*BLUE, 255))
        x += width + gap

    if abs(squeeze - 1.0) > 0.01:
        layer = layer.resize((max(1, int(layer.width * squeeze)), layer.height),
                             Image.BICUBIC)
    img.alpha_composite(layer, (int((MASTER_W - layer.width) / 2), 58))
    return {"font": Path(family).name if family else "default",
            "stroke": stroke, "gap": round(gap, 3), "squeeze": round(squeeze, 3)}


def render_clean(text: str, dept: str,
                 rng: random.Random | None = None) -> tuple[np.ndarray, dict]:
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
