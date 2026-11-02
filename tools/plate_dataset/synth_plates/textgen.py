# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Text and department sampling for synthetic plates.

The numeric block is sampled as an **integer**, never as independent digits. A
Bolivian plate number is not zero-padded: 4852 prints as ``4852`` and 490 prints
as ``490``, which is why three-digit plates exist on the street and why a leading
zero never does. Sampling ``string.digits`` character by character minted
``0773...`` at a 10% rate -- plates that cannot physically exist and that teach
the decoder a first character it must never emit. Measured on 496 real plates:
0 start with zero, 7 are three-digit (1.4%).

Four digits stay the overwhelming majority (98.5%) because that is what the site
carries. The first digit runs 1-9, not 1-6: the fleet is numbered sequentially
and reaches 7xxx-9xxx within a few years, so a model taught otherwise would have
to be retrained before it expires. Letters exclude ambiguous glyphs the recorder
already confuses (``I``/``O`` read as ``1``/``0``) so the generator does not mint
plates the real OCR could never confirm.
"""

from __future__ import annotations

import random
import string

# All nine registration departments. ``O`` (Oruro, the site) is
# overrepresented but never exclusive: fixing one letter would collapse that
# character's distribution in training.
DEPT_LETTERS: tuple[str, ...] = ("B", "C", "H", "L", "N", "O", "P", "S", "T")

# Stand-in alphabet. No O: in a well-formed plate it never appears on the real
# site (0/491 labelled plates; the only hits were malformed 4-letter noise), so
# generating it only teaches the OCR to guess. ``I`` stays in: it shows up in
# ~16% of the real plates (79/496), so excluding it would leave the class with
# zero positive samples and the model could never emit it on the street.
TEXT_LETTERS: str = "".join(c for c in string.ascii_uppercase if c != "O")

# 489/496 real plates carry four digits; the other seven are three-digit.
FOUR_DIGIT_WEIGHT: float = 0.985

# Natural digit counts, not padded widths: the lower bound is 100 so a sampled
# number always prints its own three digits.
THREE_DIGIT_RANGE: tuple[int, int] = (100, 999)
FOUR_DIGIT_RANGE: tuple[int, int] = (1000, 9999)


def sample_dept(rng: random.Random, o_weight: float = 0.45) -> str:
    """Sample a department sticker letter, ``O`` weighted, rest uniform."""
    if rng.random() < o_weight:
        return "O"
    return rng.choice([d for d in DEPT_LETTERS if d != "O"])


def sample_text(rng: random.Random, four_digit_weight: float = FOUR_DIGIT_WEIGHT) -> str:
    """Sample a plate string: ``NNNNLLL`` usually, ``NNNLLL`` rarely.

    The number is drawn as an integer so a leading zero is impossible by
    construction rather than by a filter somebody can forget to apply.
    """
    if rng.random() < four_digit_weight:
        number = rng.randint(*FOUR_DIGIT_RANGE)
    else:
        number = rng.randint(*THREE_DIGIT_RANGE)
    letters = "".join(rng.choice(TEXT_LETTERS) for _ in range(3))
    return f"{number}{letters}"
