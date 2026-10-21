"""Unit tests for the attribute vocabulary and the labelling queue.

These cover the bookkeeping that decides *what still needs a label*, which is
what makes the labeller resumable. They deliberately avoid cv2 and torch so the
suite stays fast and headless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from plate_dataset.attributes import (
    DEFAULT_COLORS,
    DEFAULT_TYPES,
    append_discard,
    append_label,
    image_path,
    iter_manifest,
    key_to_index,
    load_labelled,
    pending_samples,
    validate_label,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class TestKeyToIndex:
    def test_maps_number_row_onto_classes(self) -> None:
        assert key_to_index("1", 3) == 0
        assert key_to_index("3", 3) == 2

    def test_zero_selects_the_tenth_class(self) -> None:
        assert key_to_index("0", len(DEFAULT_COLORS)) == 9

    def test_zero_is_ignored_when_there_is_no_tenth_class(self) -> None:
        assert key_to_index("0", len(DEFAULT_TYPES)) is None

    def test_out_of_range_and_non_digits_are_ignored(self) -> None:
        assert key_to_index("4", 3) is None
        assert key_to_index("q", 10) is None
        assert key_to_index("", 10) is None

    def test_letter_key_reaches_the_eleventh_class(self) -> None:
        assert len(DEFAULT_COLORS) == 11
        assert key_to_index("q", len(DEFAULT_COLORS)) == 10


class TestValidateLabel:
    def test_accepts_declared_vocabulary(self) -> None:
        validate_label("vagoneta", "gris_plata", DEFAULT_TYPES, DEFAULT_COLORS)

    @pytest.mark.parametrize(
        ("vehicle_type", "color"),
        [("suv", "gris_plata"), ("vagoneta", "dorado")],
    )
    def test_rejects_label_outside_the_vocabulary(self, vehicle_type: str, color: str) -> None:
        with pytest.raises(ValueError):
            validate_label(vehicle_type, color, DEFAULT_TYPES, DEFAULT_COLORS)


class TestIterManifest:
    def test_skips_blank_and_truncated_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text('{"stem": "a"}\n\n{"stem": "b"}\n{"stem": "c"', encoding="utf-8")
        assert [row["stem"] for row in iter_manifest(path)] == ["a", "b"]

    def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        assert list(iter_manifest(tmp_path / "nope.jsonl")) == []


class TestQueue:
    def test_roundtrip_appends_and_reads_back(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels.jsonl"
        append_label(labels, "s1", "automovil", "blanco", DEFAULT_TYPES, DEFAULT_COLORS)
        append_label(labels, "s2", "minibus", "azul", DEFAULT_TYPES, DEFAULT_COLORS)

        stored = load_labelled(labels)
        assert set(stored) == {"s1", "s2"}
        assert stored["s2"]["color"] == "azul"
        assert stored["s2"]["source"] == "human"
        assert stored["s2"]["ts"]

    def test_relabelling_a_stem_keeps_the_last_verdict(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels.jsonl"
        append_label(labels, "s1", "automovil", "blanco", DEFAULT_TYPES, DEFAULT_COLORS)
        append_label(labels, "s1", "vagoneta", "negro", DEFAULT_TYPES, DEFAULT_COLORS)
        assert load_labelled(labels)["s1"]["type"] == "vagoneta"

    def test_pending_excludes_labelled_and_deduplicates(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write(manifest, [{"stem": "s1"}, {"stem": "s2"}, {"stem": "s1"}, {"frame": 1}])
        append_label(labels, "s1", "automovil", "blanco", DEFAULT_TYPES, DEFAULT_COLORS)

        pending = pending_samples(manifest, labels)
        assert [row["stem"] for row in pending] == ["s2"]

    def test_everything_is_pending_when_nothing_is_labelled(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _write(manifest, [{"stem": "s1"}, {"stem": "s2"}])
        pending = pending_samples(manifest, tmp_path / "absent.jsonl")
        assert len(pending) == 2

    def test_discard_removes_the_sample_from_the_queue(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        labels = tmp_path / "labels.jsonl"
        _write(manifest, [{"stem": "s1"}, {"stem": "s2"}])
        append_discard(labels, "s1", reason="unusable")

        stored = load_labelled(labels)
        assert stored["s1"]["discard"] is True
        assert stored["s1"]["color"] is None
        assert [row["stem"] for row in pending_samples(manifest, labels)] == ["s2"]


class TestImagePath:
    def test_is_derived_from_video_and_stem(self, tmp_path: Path) -> None:
        sample = {"video": "ch16_20260415", "stem": "ch16_20260415_track0003_f0000123"}
        path = image_path(tmp_path, sample)
        assert path == (tmp_path / "attributes" / "images" / "ch16_20260415"
                        / "ch16_20260415_track0003_f0000123.jpg")
