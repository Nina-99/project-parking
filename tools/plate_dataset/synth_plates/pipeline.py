# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Synthetic batch orchestration and export.

Target sizes are sampled from the real crop distribution (median 96x39 px),
never picked round: a synthetic plate born outside the recorder's range belongs
to another domain. Output reuses the extractor layout so training code cannot
tell real from synthetic — except for the manifest row, which always says so.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2

from plate_dataset.synth_plates.degrade import degrade
from plate_dataset.synth_plates.renderer import render_clean
from plate_dataset.synth_plates.textgen import sample_dept, sample_text


@dataclass
class SynthConfig:
    count: int = 1000
    seed: int = 7
    # First stem number. Lets a killed batch resume without rewriting history:
    # rerunning with the same count+seed regenerates identical files, while a
    # new start_index continues the sequence with fresh stems.
    start_index: int = 0
    # Real plate-crop distribution (px): median 96x39, spread 40-175 x 10-70.
    width_range: tuple[int, int] = (40, 175)
    width_median: int = 96
    height_range: tuple[int, int] = (10, 70)
    height_median: int = 39
    o_weight: float = 0.45
    four_digit_weight: float = 0.985
    night_prob: float = 0.35
    # Almost-intact plates: faint scratches and dirt, never damage that changes
    # a glyph. 0 disables it, so the wear arm is an ablation, not a default.
    wear_strength: float = 0.22
    # Parallel workers must not append to one file: interleaved lines would
    # corrupt the manifest. Each worker gets its own shard.
    manifest_name: str = "manifest.jsonl"

    def rng_for(self, index: int) -> random.Random:
        """Independent, reproducible random state for a single plate.

        Seeding once per batch ties a plate's appearance to the batch that made
        it, so the same index would render differently on a resume, a resplit or
        a second worker -- and a 300k set would not be reproducible from its
        parameters. Seeding from the index makes index -> plate stable, which is
        what lets twelve processes share the work.

        Stable *for a fixed generator revision*. The qualifier is not pedantry:
        commit 3a3e1cf put the letter I back into textgen, and afterwards the same
        index produced a different plate. A resume across a code change silently
        rewrites history; tools/repair_synth_plates.py exists because that is
        exactly what cost this dataset 33 files.
        """
        return random.Random(self.seed * 1_000_003 + index)


def _sample_size(rng: random.Random, lo: int, median: int, hi: int) -> int:
    """Triangular sampling around the median: realistic sizes are common."""
    return int(round(rng.triangular(lo, hi, median)))


@dataclass
class SynthPipeline:
    root: Path
    config: SynthConfig = field(default_factory=SynthConfig)

    def run(self) -> int:
        root = Path(self.root)
        img_dir = root / "plates" / "images" / "synth"
        lbl_dir = root / "plates" / "labels" / "synth"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        manifest = root / self.config.manifest_name

        written = 0
        with manifest.open("a", encoding="utf-8") as fh:
            for i in range(self.config.count):
                index = self.config.start_index + i
                rng = self.config.rng_for(index)
                text = sample_text(rng, self.config.four_digit_weight)
                dept = sample_dept(rng, self.config.o_weight)
                width = _sample_size(rng, *self.config.width_range, self.config.width_median)
                height = _sample_size(rng, *self.config.height_range, self.config.height_median)
                master, die = render_clean(text, dept, rng)
                crop, params = degrade(
                    master,
                    rng,
                    width,
                    height,
                    self.config.night_prob,
                    self.config.wear_strength,
                )
                params.update(die)
                stem = f"synth_{index:06d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop)
                (lbl_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
                fh.write(
                    json.dumps(
                        {
                            "kind": "synth",
                            "synth": True,
                            "stem": stem,
                            "label": text,
                            "dept": dept,
                            "width": width,
                            "height": height,
                            "params": params,
                            "ts": datetime.now(UTC).isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
        return written
