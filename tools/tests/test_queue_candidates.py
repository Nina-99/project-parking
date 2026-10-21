"""The candidate queue must use signals that never see the body type."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from queue_candidates import main


def _row(stem: str, track: int, frame: int, vehicle_class: int, kind: str,
         crop_hw: tuple[int, int] = (200, 300)) -> dict:
    return {
        "stem": stem, "video": "vid1", "track_id": track, "frame": frame,
        "kind": kind, "vehicle_class": vehicle_class,
        "crop_hw": list(crop_hw), "vehicle_score": 100.0,
    }


def _dataset(root: Path, rows: list[dict]) -> Path:
    (root / "attributes").mkdir(parents=True)
    (root / "attributes" / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return root


def _queue(root: Path, out: Path, extra: list[str]) -> list[dict]:
    main(["--dataset", str(root), "--target", "furgon", "--output", str(out), *extra])
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]


ROWS = [
    _row("a_track1_f10", 1, 10, 7, "entry", (200, 300)),
    _row("a_track1_f11", 1, 11, 7, "entry", (900, 1200)),
    _row("a_track2_f20", 2, 20, 7, "entry", (210, 310)),
    _row("a_track3_f30", 3, 30, 7, "pass"),        # queued only by mistake
    _row("a_track4_f40", 4, 40, 2, "entry"),       # a car, not a truck
]


def test_only_the_detector_class_and_the_trajectory_kind_are_used(tmp_path: Path) -> None:
    rows = _queue(_dataset(tmp_path / "ds", ROWS), tmp_path / "q.jsonl", [])
    stems = {row["stem"] for row in rows}

    assert "a_track2_f20" in stems
    assert len({row["track_id"] for row in rows}) == len(rows)  # one crop per track
    assert stems & {"a_track1_f10", "a_track1_f11"}
    assert "a_track3_f30" not in stems  # a passer, not an entry
    assert "a_track4_f40" not in stems  # a car for the detector


def test_one_crop_per_track_survives(tmp_path: Path) -> None:
    rows = _queue(_dataset(tmp_path / "ds", ROWS), tmp_path / "q.jsonl", [])
    assert len([row for row in rows if row["track_id"] == 1]) == 1


def test_every_row_explains_why_it_was_queued(tmp_path: Path) -> None:
    rows = _queue(_dataset(tmp_path / "ds", ROWS), tmp_path / "q.jsonl", [])
    assert all("detector_class=7(furgon)" in row["reason"] for row in rows)
    assert all("kind=entry" in row["reason"] for row in rows)


def test_limit_truncates_the_queue(tmp_path: Path) -> None:
    rows = _queue(_dataset(tmp_path / "ds", ROWS), tmp_path / "q.jsonl",
                  ["--limit", "2", "--kinds", "entry,pass"])
    assert len(rows) == 2


def test_already_labelled_frames_are_skipped(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "ds", ROWS)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"frame": "1024_rank2_frame_20"}) + "\n", encoding="utf-8")

    rows = _queue(root, tmp_path / "q.jsonl", ["--exclude-labelled", str(labels)])

    assert "a_track2_f20" not in {row["stem"] for row in rows}


def test_an_unknown_target_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--dataset", str(tmp_path), "--target", "camion", "--output", str(tmp_path / "q")])


def test_the_labeller_can_resolve_the_queue(tmp_path: Path) -> None:
    from label_type_gui import queue_images

    queue = tmp_path / "q.jsonl"
    queue.write_text(
        json.dumps({"stem": "a", "image": "attributes/images/vid/a.jpg"}) + "\n"
        + "\n"
        + json.dumps({"stem": "b", "image": None}) + "\n",
        encoding="utf-8")

    images = queue_images(queue, tmp_path)

    assert images == [tmp_path / "attributes/images/vid/a.jpg"]


MIXED = [
    {"stem": "ch6_track1_f1", "video": "XVR_ch6_main_x", "track_id": 1, "frame": 1,
     "kind": "entry", "vehicle_class": 7, "crop_hw": [200, 300], "vehicle_score": 1.0},
    {"stem": "ch6_track2_f2", "video": "XVR_ch6_main_x", "track_id": 2, "frame": 2,
     "kind": "entry", "vehicle_class": 2, "crop_hw": [200, 300], "vehicle_score": 1.0},
    {"stem": "ch16_track3_f3", "video": "C.OCCIDENTE_ch16_main_y", "track_id": 3, "frame": 3,
     "kind": "entry", "vehicle_class": 7, "crop_hw": [200, 300], "vehicle_score": 1.0},
]


def test_explicit_classes_replace_the_named_target(tmp_path: Path) -> None:
    rows = _queue(_dataset(tmp_path / "ds", MIXED), tmp_path / "q.jsonl",
                  ["--target", "furgon", "--classes", "2"])
    assert {row["vehicle_class"] for row in rows} == {2}


def test_a_video_filter_keeps_one_camera(tmp_path: Path) -> None:
    rows = _queue(_dataset(tmp_path / "ds", MIXED), tmp_path / "q.jsonl",
                  ["--target", "furgon", "--videos", "XVR_ch6"])
    assert {row["video"] for row in rows} == {"XVR_ch6_main_x"}


def test_the_reason_records_that_the_classes_were_explicit(tmp_path: Path) -> None:
    rows = _queue(_dataset(tmp_path / "ds", MIXED), tmp_path / "q.jsonl",
                  ["--classes", "2,7"])
    assert all("(custom)" in row["reason"] for row in rows)
