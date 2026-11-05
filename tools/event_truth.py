# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Ground truth for access events and how the system compares against it (H1.5).

The unit is a *crossing*, not a frame: the hypothesis asks whether the system
decides "this vehicle entered" or "this vehicle left", so each crossing is one
Bernoulli trial. Counting frames would turn one car into dozens of trials and
understate the interval.

Times are seconds of video, never wall clock. The replay harness timestamps
events with the machine clock (``SIMULATION.md`` says so), so the comparison
first shifts the system clock onto the video timeline with the instant the
replay started. Working in video seconds keeps the ground truth reusable: the
same CSV works for today's replay and for one run next month.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DIRECTIONS = ("entry", "exit")


@dataclass(frozen=True)
class Crossing:
    """One vehicle crossing one tripwire, in seconds of the source video."""

    video_seconds: float
    direction: str


@dataclass(frozen=True)
class Comparison:
    matches: list[tuple[Crossing, Crossing]]
    missed: list[Crossing]
    spurious: list[Crossing]

    @property
    def exactness(self) -> float:
        """Correct decisions over everything that should have been decided.

        TP / (TP + FP + FN): a detection added to a missed event costs the same,
        which is what an operator feels. Recall alone would let a system that
        doubles every event look perfect.
        """
        denominator = len(self.matches) + len(self.missed) + len(self.spurious)
        return len(self.matches) / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        truth = len(self.matches) + len(self.missed)
        return len(self.matches) / truth if truth else 0.0

    @property
    def precision(self) -> float:
        detected = len(self.matches) + len(self.spurious)
        return len(self.matches) / detected if detected else 0.0


def read_truth(path: Path) -> list[Crossing]:
    """Read the reviewer's CSV: ``video_seconds,direction``."""
    crossings = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            direction = row["direction"].strip().lower()
            if direction not in DIRECTIONS:
                raise ValueError(f"unknown direction {direction!r} in {path}")
            crossings.append(Crossing(float(row["video_seconds"]), direction))
    return sorted(crossings, key=lambda item: item.video_seconds)


def write_truth(path: Path, crossings: list[Crossing]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_seconds", "direction"])
        for crossing in sorted(crossings, key=lambda item: item.video_seconds):
            writer.writerow([f"{crossing.video_seconds:.3f}", crossing.direction])


def shift_onto_video(moments: list[datetime], replay_start: datetime,
                     video_start: float = 0.0) -> list[float]:
    """Wall-clock instants -> seconds of video, given when the replay began.

    The two sides are aligned before subtracting: the API may return ``Z``
    timestamps while the person typed a local one, and subtracting an aware
    datetime from a naive one raises instead of measuring.
    """
    return [video_start + (_aligned(moment, replay_start) - replay_start).total_seconds()
            for moment in moments]


def _aligned(moment: datetime, reference: datetime) -> datetime:
    if reference.tzinfo is None and moment.tzinfo is not None:
        return moment.replace(tzinfo=None)
    if reference.tzinfo is not None and moment.tzinfo is None:
        return moment.replace(tzinfo=reference.tzinfo)
    return moment


def compare(truth: list[Crossing], detected: list[Crossing],
            tolerance: float = 3.0) -> Comparison:
    """One-to-one matching by proximity, never across directions.

    An entry matched to an exit would be a coincidence of the clock, not a
    correct decision, so direction is part of the key. Pairs are taken shortest
    distance first: a greedy pass in time order can spend a detection on the
    wrong neighbour and turn two correct decisions into one.
    """
    candidates = []
    for truth_index, expected in enumerate(truth):
        for detected_index, found in enumerate(detected):
            if expected.direction != found.direction:
                continue
            distance = abs(expected.video_seconds - found.video_seconds)
            if distance <= tolerance:
                candidates.append((distance, truth_index, detected_index))
    candidates.sort()

    used_truth: set[int] = set()
    used_detected: set[int] = set()
    matches: list[tuple[Crossing, Crossing]] = []
    for _, truth_index, detected_index in candidates:
        if truth_index in used_truth or detected_index in used_detected:
            continue
        used_truth.add(truth_index)
        used_detected.add(detected_index)
        matches.append((truth[truth_index], detected[detected_index]))

    missed = [item for index, item in enumerate(truth) if index not in used_truth]
    spurious = [item for index, item in enumerate(detected) if index not in used_detected]
    return Comparison(matches=matches, missed=missed, spurious=spurious)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    import math

    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))
