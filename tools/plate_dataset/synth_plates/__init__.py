# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Synthetic Bolivian license-plate generator.

Renders the national format (1999-present) from text + department letter and
degrades it through the physical pipeline — geometry, light, sensor — so the
samples land in the same distribution as the real crops (median ~96x39 px).

Layout decisions mirror the verified domain: blue on white, ``BOLIVIA`` on top,
department sticker top-right, main text ``NNN(N)LLL``. The font is a condensed
bold stand-in, not the official die: shape and layout match, strokes do not.
"""

from plate_dataset.synth_plates.ablation import (
    OcrScores,
    cer,
    collect_real_plates,
    levenshtein,
    score_pairs,
    split_by_video,
)
from plate_dataset.synth_plates.degrade import degrade
from plate_dataset.synth_plates.pipeline import SynthPipeline
from plate_dataset.synth_plates.renderer import render_clean
from plate_dataset.synth_plates.textgen import sample_dept, sample_text

__all__ = [
    "OcrScores",
    "SynthPipeline",
    "cer",
    "collect_real_plates",
    "degrade",
    "levenshtein",
    "render_clean",
    "sample_dept",
    "sample_text",
    "score_pairs",
    "split_by_video",
]
