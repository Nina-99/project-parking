"""Text and department sampling for synthetic plates.

The format split is a domain decision, not a guess: 90% ``NNNNLLL`` (current
fleet) and 10% ``NNNLLL`` (older vehicles still seen on site). Letters exclude
ambiguous glyphs the recorder already confuses (``I``/``O`` read as ``1``/``0``)
so the generator does not mint plates the real OCR could never confirm.
"""

from __future__ import annotations

import random
import string

# All nine registration departments. ``O`` (Oruro, the site) is
# overrepresented but never exclusive: fixing one letter would collapse that
# character's distribution in training.
DEPT_LETTERS: tuple[str, ...] = ("B", "C", "H", "L", "N", "O", "P", "S", "T")

# Stand-in alphabet. No I/O: at 40-130 px wide they are indistinguishable from
# 1/0 on the recorder, so generating them only teaches the OCR to guess.
TEXT_LETTERS: str = "".join(c for c in string.ascii_uppercase if c not in "IO")

FOUR_DIGIT_WEIGHT: float = 0.90


def sample_dept(rng: random.Random, o_weight: float = 0.45) -> str:
    """Sample a department sticker letter, ``O`` weighted, rest uniform."""
    if rng.random() < o_weight:
        return "O"
    return rng.choice([d for d in DEPT_LETTERS if d != "O"])


def sample_text(rng: random.Random, four_digit_weight: float = FOUR_DIGIT_WEIGHT) -> str:
    """Sample a plate string: ``NNNNLLL`` or, rarely, ``NNNLLL``."""
    digits = 4 if rng.random() < four_digit_weight else 3
    number = "".join(rng.choice(string.digits) for _ in range(digits))
    letters = "".join(rng.choice(TEXT_LETTERS) for _ in range(3))
    return f"{number}{letters}"
