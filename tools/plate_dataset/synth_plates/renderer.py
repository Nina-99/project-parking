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

_BOLD_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
_CONDENSED_CANDIDATES = (
    "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSansNarrow-Bold.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _CONDENSED_CANDIDATES + _BOLD_CANDIDATES:
        if Path(path).exists():
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


def render_clean(text: str, dept: str) -> np.ndarray:
    """Draw the sharp master (BGR, ``MASTER_W`` x ``MASTER_H``)."""
    img = Image.new("RGB", (MASTER_W, MASTER_H), WHITE)
    draw = ImageDraw.Draw(img)

    # Outer frame.
    draw.rectangle([4, 4, MASTER_W - 5, MASTER_H - 5], outline=BLUE, width=6)

    # Header: tricolor bar + BOLIVIA + department sticker box.
    draw.rectangle([18, 14, 58, 44], fill=(214, 40, 40))
    draw.rectangle([18, 24, 58, 34], fill=(255, 235, 59))
    draw.rectangle([18, 29, 58, 44], fill=(56, 142, 60))
    _centered(draw, MASTER_W / 2.0 - 30, 12, "BOLIVIA", _load_font(30), BLUE)
    draw.rectangle([MASTER_W - 78, 12, MASTER_W - 18, 46], outline=BLUE, width=3)
    _centered(draw, MASTER_W - 48, 14, dept, _load_font(26), BLUE)

    # Main text, centered in the remaining band. Seven characters get a smaller
    # cut so they never touch the frame: the recorder already clips edges.
    _centered(draw, MASTER_W / 2.0, 62, text, _load_font(96 if len(text) <= 6 else 82),
              BLUE)

    return np.asarray(img)[:, :, ::-1].copy()
