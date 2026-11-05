# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: Proprietary

"""The H1.5 comparison: matching, direction and the definition of exactness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from event_truth import (
    Comparison,
    Crossing,
    compare,
    read_truth,
    shift_onto_video,
    wilson,
    write_truth,
)


def test_a_detection_within_tolerance_matches() -> None:
    truth = [Crossing(10.0, "entry")]
    detected = [Crossing(11.5, "entry")]
    result = compare(truth, detected, tolerance=3.0)
    assert len(result.matches) == 1
    assert result.missed == []
    assert result.spurious == []


def test_a_detection_beyond_tolerance_does_not_match() -> None:
    result = compare([Crossing(10.0, "entry")], [Crossing(20.0, "entry")], tolerance=3.0)
    assert result.matches == []
    assert len(result.missed) == 1
    assert len(result.spurious) == 1


def test_an_entry_never_matches_an_exit() -> None:
    """Same instant, opposite direction: a clock coincidence, not a decision."""
    result = compare([Crossing(10.0, "entry")], [Crossing(10.0, "exit")], tolerance=3.0)
    assert result.matches == []
    assert len(result.missed) == 1
    assert len(result.spurious) == 1


def test_matching_is_one_to_one() -> None:
    """Two detections cannot both be credited to one ground-truth crossing."""
    truth = [Crossing(10.0, "entry")]
    detected = [Crossing(10.2, "entry"), Crossing(10.4, "entry")]
    result = compare(truth, detected, tolerance=3.0)
    assert len(result.matches) == 1
    assert len(result.spurious) == 1


def test_the_nearest_pair_wins_over_the_earliest() -> None:
    """Greedy in time order would spend the first detection on the wrong car."""
    truth = [Crossing(10.0, "entry"), Crossing(10.5, "entry")]
    detected = [Crossing(12.9, "entry"), Crossing(10.2, "entry")]
    result = compare(truth, detected, tolerance=3.0)
    assert len(result.matches) == 2
    assert result.spurious == []


def test_exactness_penalises_a_doubled_event_like_a_missed_one() -> None:
    # One correct, one missed, one spurious: 1 / 3.
    comparison = Comparison(
        matches=[(Crossing(1.0, "entry"), Crossing(1.0, "entry"))],
        missed=[Crossing(5.0, "entry")],
        spurious=[Crossing(9.0, "entry")],
    )
    assert comparison.exactness == pytest.approx(1 / 3)
    assert comparison.recall == pytest.approx(1 / 2)
    assert comparison.precision == pytest.approx(1 / 2)


def test_exactness_of_an_empty_comparison_is_zero_not_an_error() -> None:
    assert Comparison([], [], []).exactness == 0.0


def test_truth_round_trips_through_csv(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    write_truth(path, [Crossing(3.5, "exit"), Crossing(1.25, "entry")])
    assert read_truth(path) == [Crossing(1.25, "entry"), Crossing(3.5, "exit")]


def test_an_unknown_direction_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    path.write_text("video_seconds,direction\n1.0,sideways\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_truth(path)


def test_wall_clock_is_shifted_onto_the_video_timeline() -> None:
    start = datetime(2026, 11, 2, 21, 0, 0, tzinfo=UTC)
    moments = [start + timedelta(seconds=12), start + timedelta(seconds=90)]
    assert shift_onto_video(moments, start) == [12.0, 90.0]
    assert shift_onto_video(moments, start, video_start=600.0) == [612.0, 690.0]


def test_wilson_brackets_the_proportion() -> None:
    low, high = wilson(9, 10)
    assert 0.5 < low < 0.9 < high <= 1.0


def test_the_speed_ladder_steps_and_stops_at_the_ends() -> None:
    from label_events_gui import _next_speed

    assert _next_speed(1.0, faster=True) == 1.5
    assert _next_speed(4.0, faster=True) == 4.0
    assert _next_speed(0.5, faster=False) == 0.5
    assert _next_speed(2.0, faster=False) == 1.5
    # A speed that is not on the ladder snaps to the nearest one first.
    assert _next_speed(1.7, faster=True) == 2.0


def test_an_aware_api_timestamp_against_a_naive_replay_start() -> None:
    """The API may return Z; a person types a local time. Subtracting would raise."""
    start = datetime(2026, 11, 2, 21, 0, 0)
    moment = datetime(2026, 11, 2, 21, 0, 12, tzinfo=UTC)
    assert shift_onto_video([moment], start) == [12.0]
