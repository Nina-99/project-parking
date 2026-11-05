# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Contract tests for the plate-text prefill.

Two properties are load-bearing. A prefill must never become a silent label: the
row records whether the text was typed or accepted as-is, because a pseudo-label
with a human signature is still a pseudo-label. And a bad proposal must not
prefill a line the reviewer has to erase before starting.
"""

from __future__ import annotations

import json
from pathlib import Path

from label_plate_text_gui import build_parser, label_source, load_proposals


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class TestProposals:
    def test_reads_text_by_stem(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "proposed.jsonl",
            [
                {"stem": "a", "text": "4852phd", "box": [1, 2, 3, 4]},
                {"stem": "b", "text": "490IRB"},
            ],
        )
        assert load_proposals(path) == {"a": "4852PHD", "b": "490IRB"}

    def test_skips_proposals_that_cannot_be_plates(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "proposed.jsonl",
            [
                {"stem": "empty", "text": None},
                {"stem": "blank", "text": "   "},
                {"stem": "spaced", "text": " 4852 PHD "},
                {"stem": "symbols", "text": "4852-PHD"},
                {"stem": "accented", "text": "4852PHD\u00d1"},
                {"stem": "good", "text": "4852PHD"},
            ],
        )
        # "spaced" collapses to a valid plate; the rest would prefill garbage.
        assert load_proposals(path) == {"spaced": "4852PHD", "good": "4852PHD"}

    def test_last_row_wins(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "proposed.jsonl",
            [
                {"stem": "a", "text": "1111AAA"},
                {"stem": "a", "text": "2222BBB"},
            ],
        )
        assert load_proposals(path) == {"a": "2222BBB"}

    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_proposals(tmp_path / "nope.jsonl") == {}
        assert load_proposals(None) == {}


class TestProvenance:
    def test_accepting_the_suggestion_is_not_a_human_label(self) -> None:
        assert label_source("4852PHD", edited=False) == "paddle"

    def test_any_edit_makes_it_human(self) -> None:
        assert label_source("4852PHD", edited=True) == "human"

    def test_typing_from_scratch_is_human(self) -> None:
        assert label_source("", edited=True) == "human"
        assert label_source("", edited=False) == "human"


class TestCli:
    def test_default_is_every_pending_crop(self) -> None:
        # The old default of 100 made a truncated dataset look finished.
        args = build_parser().parse_args(["--verified", "v.jsonl", "--output", "t.jsonl"])
        assert args.limit == 0
        assert args.proposals is None
