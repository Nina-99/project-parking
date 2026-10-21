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
import numpy as np

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
    four_digit_weight: float = 0.90
    night_prob: float = 0.35


def _sample_size(rng: random.Random, lo: int, median: int, hi: int) -> int:
    """Triangular sampling around the median: realistic sizes are common."""
    return int(round(rng.triangular(lo, hi, median)))


@dataclass
class SynthPipeline:
    root: Path
    config: SynthConfig = field(default_factory=SynthConfig)

    def run(self) -> int:
        rng = random.Random(self.config.count * 100003 + self.config.seed)
        np.random.seed(self.config.seed + self.config.start_index)
        root = Path(self.root)
        img_dir = root / "plates" / "images" / "synth"
        lbl_dir = root / "plates" / "labels" / "synth"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.jsonl"

        written = 0
        with manifest.open("a", encoding="utf-8") as fh:
            for i in range(self.config.count):
                text = sample_text(rng, self.config.four_digit_weight)
                dept = sample_dept(rng, self.config.o_weight)
                width = _sample_size(rng, *self.config.width_range, self.config.width_median)
                height = _sample_size(rng, *self.config.height_range, self.config.height_median)
                master = render_clean(text, dept)
                crop, params = degrade(master, rng, width, height, self.config.night_prob)
                stem = f"synth_{self.config.start_index + i:06d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop)
                (lbl_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
                fh.write(json.dumps({
                    "kind": "synth",
                    "synth": True,
                    "stem": stem,
                    "label": text,
                    "dept": dept,
                    "width": width,
                    "height": height,
                    "params": params,
                    "ts": datetime.now(UTC).isoformat(),
                }, ensure_ascii=False) + "\n")
                written += 1
        return written
