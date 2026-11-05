# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Unit tests for the synthetic plate generator.

Fast and headless: small masters, few samples, fixed seeds. What matters here
is the contract — layout identical to the extractor, manifest always traceable,
sizes inside the real distribution — not the aesthetics of any single crop.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
from plate_dataset.synth_plates.degrade import degrade
from plate_dataset.synth_plates.pipeline import SynthConfig, SynthPipeline
from plate_dataset.synth_plates.renderer import MASTER_H, MASTER_W, render_clean
from plate_dataset.synth_plates.textgen import (
    DEPT_LETTERS,
    sample_dept,
    sample_text,
)


class TestTextgen:
    def test_four_digit_format_dominates(self) -> None:
        # 489/496 real plates carry four digits, so the generator must not drift
        # back to the old 90/10 split.
        rng = random.Random(0)
        texts = [sample_text(rng) for _ in range(1000)]
        four = sum(1 for t in texts if len(t) == 7)
        assert 970 <= four <= 998

    def test_three_digit_format_exists_but_rare(self) -> None:
        # 7/496 real plates are three-digit: 1.4%, not the 10% it used to mint.
        rng = random.Random(1)
        texts = [sample_text(rng) for _ in range(2000)]
        three = sum(1 for t in texts if len(t) == 6)
        assert 10 <= three <= 55

    def test_never_starts_with_zero(self) -> None:
        # A Bolivian plate number is not zero-padded: 490 prints as 490. A
        # leading zero is a plate that cannot exist, and the decoder would learn
        # to emit it as an insertion that costs the whole exact-match.
        rng = random.Random(4)
        texts = [sample_text(rng) for _ in range(5000)]
        assert not any(t.startswith("0") for t in texts)

    def test_first_digit_reaches_nine(self) -> None:
        # The fleet is numbered sequentially and will reach 7xxx-9xxx within a
        # few years; capping at the range observed today would expire the model.
        rng = random.Random(5)
        first = {sample_text(rng)[0] for _ in range(5000)}
        assert first == set("123456789")

    def test_three_digit_numbers_stay_three_digits(self) -> None:
        rng = random.Random(6)
        three = [t for t in (sample_text(rng, 0.0) for _ in range(500)) if len(t) == 6]
        assert three, "the three-digit branch must be reachable"
        assert all(t[0] != "0" for t in three)
        assert all(len(t) == 6 for t in three)

    def test_only_o_is_excluded(self) -> None:
        rng = random.Random(2)
        texts = [sample_text(rng) for _ in range(500)]
        # O never appears on real well-formed plates, so it is never generated.
        assert all("O" not in t for t in texts)
        # I is common on the street (~16% of real plates), so it must be sampled.
        assert any("I" in t for t in texts)

    def test_dept_covers_all_nine_with_o_majority(self) -> None:
        rng = random.Random(3)
        depts = [sample_dept(rng) for _ in range(2000)]
        assert set(depts) == set(DEPT_LETTERS)
        o_share = sum(1 for d in depts if d == "O") / len(depts)
        assert 0.35 <= o_share <= 0.55


class TestRenderer:
    def test_master_size_and_channels(self) -> None:
        master, _ = render_clean("4852PHD", "O")
        assert master.shape == (MASTER_H, MASTER_W, 3)

    def test_dept_changes_pixels(self) -> None:
        a, _ = render_clean("4852PHD", "O")
        b, _ = render_clean("4852PHD", "L")
        assert (a != b).any()


class TestDegrade:
    def test_output_matches_target_size(self) -> None:
        rng = random.Random(4)
        master, _ = render_clean("4852PHD", "O")
        crop, params = degrade(master, rng, 96, 39)
        assert crop.shape[1] == 96 and crop.shape[0] == 39
        assert params["target_hw"] == [39, 96]

    def test_deterministic_under_seed(self) -> None:
        def run() -> bytes:
            rng = random.Random(11)
            master, _ = render_clean("4852PHD", "O")
            crop, _ = degrade(master, rng, 80, 30)
            _, buf = cv2.imencode(".png", crop)
            return buf.tobytes()

        assert run() == run()


class TestPipeline:
    def test_writes_layout_and_manifest(self, tmp_path: Path) -> None:
        written = SynthPipeline(tmp_path, SynthConfig(count=5, seed=0)).run()
        assert written == 5
        assert len(list((tmp_path / "plates" / "images" / "synth").glob("*.jpg"))) == 5
        labels = sorted((tmp_path / "plates" / "labels" / "synth").glob("*.txt"))
        assert len(labels) == 5
        rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
        assert len(rows) == 5
        assert all(row["synth"] is True and row["kind"] == "synth" for row in rows)
        assert labels[0].read_text().strip() == rows[0]["label"]
