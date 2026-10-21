"""Tests for the ablation split and OCR metrics.

The split tests are the load-bearing ones: if a video ever lands on both
sides, the ablation measures memorisation instead of generalisation and H1.4
falls over.
"""

from __future__ import annotations

from pathlib import Path

from plate_dataset.synth_plates.ablation import (
    cer,
    collect_real_plates,
    levenshtein,
    score_pairs,
    split_by_video,
)


class TestLevenshtein:
    def test_identical_is_zero(self) -> None:
        assert levenshtein("4852PHD", "4852PHD") == 0

    def test_single_substitution(self) -> None:
        assert levenshtein("4852PHD", "4852PHC") == 1

    def test_empty_hypothesis_costs_full_length(self) -> None:
        assert levenshtein("4852PHD", "") == 7

    def test_empty_reference_scores_zero_distance(self) -> None:
        assert levenshtein("", "") == 0


class TestCer:
    def test_perfect_read_scores_zero(self) -> None:
        assert cer("4852PHD", "4852PHD") == 0.0

    def test_one_error_in_seven(self) -> None:
        assert abs(cer("4852PHD", "4852PHC") - 1 / 7) < 1e-9


class TestScorePairs:
    def test_thesis_criteria_on_toy_pairs(self) -> None:
        scores = score_pairs([("4852PHD", "4852PHD"), ("4852PHD", "4852PHC")])
        assert scores.n == 2
        assert scores.exact_match == 0.5
        assert abs(scores.mean_cer - (1 / 7) / 2) < 1e-9
        assert abs(scores.char_accuracy - (1 - (1 / 7) / 2)) < 1e-9

    def test_empty_scores_zero(self) -> None:
        scores = score_pairs([])
        assert scores.n == 0 and scores.exact_match == 0.0


class TestSplitByVideo:
    def test_no_video_on_both_sides(self) -> None:
        by_video = {f"video{i}": [f"s{i}_{j}" for j in range(5)] for i in range(10)}
        train, test = split_by_video(by_video, test_fraction=0.2, seed=7)
        train_videos = {s.rsplit("_", 2)[0] for s in train}
        test_videos = {s.rsplit("_", 2)[0] for s in test}
        assert not (train_videos & test_videos)
        assert len(train) + len(test) == 50

    def test_deterministic_across_runs(self) -> None:
        by_video = {f"video{i}": [f"s{i}"] for i in range(20)}
        assert split_by_video(by_video) == split_by_video(by_video)

    def test_collect_skips_synth(self, tmp_path: Path) -> None:
        img = tmp_path / "plates" / "images"
        lbl = tmp_path / "plates" / "labels"
        (img / "videoA").mkdir(parents=True)
        (lbl / "videoA").mkdir(parents=True)
        (img / "videoA" / "a.jpg").write_bytes(b"x")
        (lbl / "videoA" / "a.txt").write_text("4852PHD\n", encoding="utf-8")
        (img / "synth").mkdir(parents=True)
        (img / "synth" / "s.jpg").write_bytes(b"x")
        collected = collect_real_plates(tmp_path)
        assert list(collected) == ["videoA"]
