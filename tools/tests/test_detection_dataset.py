# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The parts of the H1.1 toolchain that can be wrong silently.

Geometry, label round-trips, greedy matching and AP are pure: they decide the
number the thesis reports, so they are tested without a model or a window. A bug
in any of them produces a plausible-looking mAP that means nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from detection_dataset import (
    ImageResult,
    Prediction,
    average_precision,
    bootstrap_ap,
    from_yolo,
    iou,
    label_path_for,
    match,
    match_per_prediction,
    read_labels,
    to_yolo,
    wilson,
    write_labels,
)
from label_boxes_gui import Viewport, is_blind, load_proposals, was_modified


def test_iou_is_one_for_identical_boxes_and_zero_when_apart() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_matches_the_hand_computed_overlap() -> None:
    # 8x8 overlap over a union of 100 + 100 - 64 = 136.
    assert iou((0, 0, 10, 10), (2, 2, 12, 12)) == pytest.approx(64 / 136)


def test_a_label_round_trips_through_yolo_coordinates() -> None:
    box = (100, 50, 300, 250)
    normalised = to_yolo(box, 1280, 720)
    assert from_yolo(normalised, 1280, 720) == box


def test_written_labels_can_be_read_back(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "a.txt"
    write_labels(path, [(10, 20, 110, 220)], 1280, 720)
    assert read_labels(path, 1280, 720) == [(10, 20, 110, 220)]


def test_a_negative_is_an_empty_file_not_a_missing_one(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    write_labels(path, [], 1280, 720)
    assert path.exists()
    assert read_labels(path, 1280, 720) == []


def test_a_missing_label_file_is_an_error() -> None:
    with pytest.raises(FileNotFoundError):
        read_labels(Path("/nonexistent/labels/a.txt"), 1280, 720)


def test_two_boxes_on_one_vehicle_cost_a_false_positive() -> None:
    truth = [(100, 100, 300, 300)]
    predictions = [
        Prediction((100, 100, 300, 300), 0.9),
        Prediction((102, 102, 302, 302), 0.8),
    ]
    assert match_per_prediction(predictions, truth, 0.5) == [True, False]
    assert match(predictions, truth, 0.5) == (1, 1, 0)


def test_a_missed_vehicle_is_a_false_negative() -> None:
    assert match([], [(0, 0, 10, 10)], 0.5) == (0, 0, 1)


def test_ap_uses_labelled_vehicles_as_the_recall_denominator() -> None:
    """One hit of two labelled vehicles is recall 0.5, not 1.0.

    Dividing by the number of hits instead of the number of labels is the
    mistake that makes a detector with poor recall look perfect.
    """
    results = [ImageResult(scored=[(0.9, True)], positives=2)]
    assert average_precision(results) == pytest.approx(0.5)


def test_ap_is_one_when_every_vehicle_is_found_before_any_error() -> None:
    results = [
        ImageResult(scored=[(0.9, True), (0.8, True)], positives=2),
        ImageResult(scored=[(0.7, True)], positives=1),
    ]
    assert average_precision(results) == pytest.approx(1.0)


def test_ap_is_zero_without_predictions() -> None:
    assert average_precision([ImageResult(scored=[], positives=3)]) == 0.0


def test_false_positives_after_the_last_hit_do_not_lower_ap() -> None:
    """A detector is not punished for the confidence values below its recall.

    This is the regression test for sorting the curve by recall: every point
    after the last hit has recall 1.0, and reordering them by precision would
    integrate the *worst* precision instead of the best.
    """
    results = [ImageResult(
        scored=[(0.9, True)] + [(0.1 - i * 0.001, False) for i in range(50)],
        positives=1,
    )]
    assert average_precision(results) == pytest.approx(1.0)


def test_ap_interpolates_between_two_hits() -> None:
    # hit at precision 1.0, then a false positive, then the second hit at 2/3:
    # 0.5 * 1.0 + 0.5 * 2/3.
    results = [ImageResult(
        scored=[(0.9, True), (0.8, False), (0.7, True)], positives=2)]
    assert average_precision(results) == pytest.approx(0.5 + 0.5 * (2 / 3))


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    results = [
        ImageResult(scored=[(0.9, True), (0.4, False)], positives=1),
        ImageResult(scored=[(0.8, True)], positives=1),
        ImageResult(scored=[(0.6, True)], positives=1),
    ]
    low, high = bootstrap_ap(results, rounds=300, seed=1)
    assert 0.0 <= low <= average_precision(results) <= high <= 1.0


def test_wilson_stays_inside_the_unit_interval_at_the_extremes() -> None:
    low, high = wilson(0, 5)
    assert 0.0 <= low < high <= 1.0
    low, high = wilson(5, 5)
    assert 0.0 <= low < high <= 1.0


def test_label_path_mirrors_the_image_subdirectory(tmp_path: Path) -> None:
    image = tmp_path / "images" / "channel16" / "a.jpg"
    assert label_path_for(image, tmp_path / "images", tmp_path / "labels") == (
        tmp_path / "labels" / "channel16" / "a.txt"
    )


def test_viewport_maps_a_click_back_to_frame_pixels() -> None:
    viewport = Viewport(1280, 720)
    x, y = viewport.to_original(*viewport.to_view((100, 50, 300, 250))[:2])
    assert abs(x - 100) <= 2
    assert abs(y - 50) <= 2


def test_viewport_never_scales_up_a_small_frame() -> None:
    viewport = Viewport(320, 200)
    assert viewport.scale == 1.0


def test_proposals_are_keyed_by_the_path_inside_images(tmp_path: Path) -> None:
    (tmp_path / "proposals.jsonl").write_text(
        '{"image": "channel16/a.jpg", "boxes": [[1, 2, 3, 4]]}\n', encoding="utf-8")
    assert load_proposals(tmp_path) == {"channel16/a.jpg": [(1, 2, 3, 4)]}


def test_proposals_are_empty_when_the_file_is_absent(tmp_path: Path) -> None:
    assert load_proposals(tmp_path) == {}


def test_the_blind_subset_is_stable_and_keyed_on_the_file_name() -> None:
    names = [f"channel6/video_{i}.jpg" for i in range(200)]
    chosen = [name for name in names if is_blind(name, 4)]
    again = [name for name in names if is_blind(name, 4)]
    assert chosen == again, "the same files must be blind on every session"
    assert 35 <= len(chosen) <= 65, f"one in four expected, got {len(chosen)}"
    # Adding footage later must not change which images were blind.
    extended = [name for name in names + ["channel6/new.jpg"] if is_blind(name, 4)]
    assert extended == chosen


def test_blind_is_disabled_at_zero_and_one() -> None:
    assert not any(is_blind(f"a{i}.jpg", 0) for i in range(50))
    assert not any(is_blind(f"a{i}.jpg", 1) for i in range(50))


def test_an_untouched_proposal_is_not_counted_as_modified() -> None:
    box = (100, 50, 300, 250)
    assert was_modified([box], [box]) is False
    assert was_modified([(101, 51, 301, 251)], [box]) is False


def test_a_deleted_proposal_counts_as_modified() -> None:
    assert was_modified([], [(100, 50, 300, 250)]) is True
    assert was_modified([(1, 2, 300, 400)], [(100, 50, 300, 250)]) is True


def test_the_evaluator_refuses_to_score_an_unlabelled_set(tmp_path: Path) -> None:
    """A missing label is not "no vehicles": it is work nobody did."""
    import sys

    from eval_vehicle_detection import main as evaluate

    (tmp_path / "images" / "channel16").mkdir(parents=True)
    (tmp_path / "images" / "channel16" / "a.jpg").write_bytes(b"not really a jpeg")

    argv = ["--root", str(tmp_path), "--model", "yolo26n.pt"]
    original = sys.argv
    sys.argv = ["eval_vehicle_detection.py", *argv]
    try:
        with pytest.raises(SystemExit) as exit_info:
            evaluate(argv)
    finally:
        sys.argv = original
    assert "have no label file" in str(exit_info.value)
