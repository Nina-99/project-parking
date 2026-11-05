# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The repair tool must find what is broken and never guess which one it is.

Generation is expensive, so these tests stop at detection: what counts as
corrupt, what counts as missing, and what the verification rule is when the
original label still exists.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from repair_synth_plates import ALPHABETS, jpeg_ok, scan


def _jpeg() -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.zeros((12, 12, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def _root(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "plates" / "images" / "synth"
    labels = tmp_path / "plates" / "labels" / "synth"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    return tmp_path, images, labels


def test_a_truncated_file_is_corrupt_and_a_missing_stem_is_missing(tmp_path: Path) -> None:
    root, images, labels = _root(tmp_path)
    good = _jpeg()
    (images / "synth_000000.jpg").write_bytes(good)
    (labels / "synth_000000.txt").write_text("1234ABC\n", encoding="utf-8")

    (images / "synth_000001.jpg").write_bytes(good[: len(good) // 2])
    (labels / "synth_000001.txt").write_text("5678DEF\n", encoding="utf-8")
    # synth_000002 does not exist at all

    targets = scan(root, 3, force=False)

    reasons = {target.stem: target.reason for target in targets}
    assert reasons == {"synth_000001": "corrupt", "synth_000002": "missing"}
    expected = {target.stem: target.expected for target in targets}
    assert expected["synth_000001"] == "5678DEF"
    assert expected["synth_000002"] is None


def test_a_valid_file_is_not_touched(tmp_path: Path) -> None:
    root, images, labels = _root(tmp_path)
    (images / "synth_000000.jpg").write_bytes(_jpeg())
    (labels / "synth_000000.txt").write_text("1234ABC\n", encoding="utf-8")

    assert scan(root, 1, force=False) == []


def test_force_targets_every_expected_stem(tmp_path: Path) -> None:
    root, images, labels = _root(tmp_path)
    (images / "synth_000000.jpg").write_bytes(_jpeg())
    (labels / "synth_000000.txt").write_text("1234ABC\n", encoding="utf-8")

    targets = scan(root, 1, force=True)

    assert [target.stem for target in targets] == ["synth_000000"]


def test_the_two_alphabets_differ_only_by_the_letter_i() -> None:
    """The repair picks a revision by matching labels; that only works if the
    alphabets really are two distinct populations."""
    assert "I" in ALPHABETS["current"]
    assert "I" not in ALPHABETS["pre-I"]
    assert set(ALPHABETS["pre-I"]) | {"I"} == set(ALPHABETS["current"])


def test_jpeg_ok_rejects_a_missing_file(tmp_path: Path) -> None:
    assert not jpeg_ok(tmp_path / "nope.jpg")
