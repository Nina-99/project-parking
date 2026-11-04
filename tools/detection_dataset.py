# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Labels, matching and intervals for the vehicle-detection measurement (H1.1).

Pure functions only: no OpenCV window, no YOLO, no filesystem traversal beyond
what the caller asks for. The evaluator and the labelling GUI share this module
so that what the reviewer saves is exactly what the evaluator reads.

One class on purpose. H1.1 asks whether a vehicle is found, not which body type
it has; the taxonomy is H1.2 and is measured on crops, where the vehicle already
fills the frame. Adding five classes here would make the annotation ambiguous
(a van at the edge of the gate is not obviously a van) without answering a
question the thesis asks.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

Box = tuple[int, int, int, int]
NormalisedBox = tuple[float, float, float, float]

DEFAULT_CLASS = 0


@dataclass(frozen=True)
class GroundTruth:
    """One labelled vehicle, in pixels of the original frame."""

    box: Box
    image: str


@dataclass(frozen=True)
class Prediction:
    """One detected vehicle, with the confidence the model gave it."""

    box: Box
    confidence: float


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two pixel boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def to_yolo(box: Box, width: int, height: int) -> NormalisedBox:
    """Pixel box -> (cx, cy, w, h) normalised, the format Ultralytics reads."""
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def from_yolo(box: NormalisedBox, width: int, height: int) -> Box:
    """(cx, cy, w, h) normalised -> pixel box, clamped to the frame."""
    cx, cy, w, h = box
    x1 = round((cx - w / 2.0) * width)
    y1 = round((cy - h / 2.0) * height)
    x2 = round((cx + w / 2.0) * width)
    y2 = round((cy + h / 2.0) * height)
    return (
        max(0, min(x1, width)), max(0, min(y1, height)),
        max(0, min(x2, width)), max(0, min(y2, height)),
    )


def write_labels(path: Path, boxes: list[Box], width: int, height: int,
                 class_index: int = DEFAULT_CLASS) -> None:
    """Write one YOLO label file. An image with no vehicles is an empty file.

    The empty file is written on purpose: it is the negative sample that stops
    the detector from being scored only on frames where something was found.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        cx, cy, w, h = to_yolo(box, width, height)
        lines.append(f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_labels(path: Path, width: int, height: int) -> list[Box]:
    """Read a YOLO label file. A missing file is an error, not an empty frame.

    The distinction matters: a missing file means the image was never reviewed,
    and scoring it as "no vehicles" would reward the detector for the reviewer's
    unfinished work.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: the image was never labelled")
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        boxes.append(from_yolo((float(cx), float(cy), float(w), float(h)), width, height))
    return boxes


def match_per_prediction(predictions: list[Prediction], ground_truth: list[Box],
                         iou_threshold: float) -> list[bool]:
    """Hit or miss for every prediction, greedy by confidence.

    Each ground-truth box is consumed at most once: two boxes on the same vehicle
    count as one hit and one false positive, which is what the system does too --
    a duplicated box inflates occupancy.
    """
    remaining = list(ground_truth)
    outcome = []
    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        best_index, best_iou = -1, iou_threshold
        for index, truth in enumerate(remaining):
            score = iou(prediction.box, truth)
            if score >= best_iou:
                best_index, best_iou = index, score
        if best_index >= 0:
            remaining.pop(best_index)
            outcome.append(True)
        else:
            outcome.append(False)
    return outcome


def match(predictions: list[Prediction], ground_truth: list[Box],
          iou_threshold: float) -> tuple[int, int, int]:
    """(true positives, false positives, false negatives) for one image."""
    outcome = match_per_prediction(predictions, ground_truth, iou_threshold)
    true_positives = sum(1 for hit in outcome if hit)
    return true_positives, len(outcome) - true_positives, len(ground_truth) - true_positives


@dataclass(frozen=True)
class ImageResult:
    """What one image contributes to the precision-recall curve.

    ``positives`` is carried separately because the recall denominator is the
    number of labelled vehicles, not the number of hits. Using the hits as the
    denominator is the classic way to inflate AP on a detector that misses a lot.
    """

    scored: list[tuple[float, bool]]
    positives: int


def average_precision(results: list[ImageResult]) -> float:
    """Area under the precision-recall curve, all-point interpolation."""
    positives = sum(result.positives for result in results)
    scored = [item for result in results for item in result.scored]
    if not scored or positives == 0:
        return 0.0
    ordered = sorted(scored, key=lambda item: item[0], reverse=True)
    tp = fp = 0
    recalls: list[float] = [0.0]
    precisions: list[float] = [0.0]
    for _, hit in ordered:
        if hit:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / positives)
        precisions.append(tp / (tp + fp))
    # All-point interpolation: make precision monotone from the right, then
    # integrate. Sorting the curve by recall before integrating would reorder
    # equal-recall points by precision and pick the worst one.
    for index in range(len(precisions) - 1, 0, -1):
        precisions[index - 1] = max(precisions[index - 1], precisions[index])
    return sum(
        (recalls[index] - recalls[index - 1]) * precisions[index]
        for index in range(1, len(recalls))
    )


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Honest at small n, unlike the normal approximation."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ap(results: list[ImageResult], rounds: int = 2000,
                 seed: int = 7) -> tuple[float, float]:
    """Percentile CI for AP by resampling whole images.

    Images, not boxes: two boxes in the same frame are not independent, and
    resampling boxes would produce an interval narrower than the data allows.
    """
    if not results:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    for _ in range(rounds):
        drawn = [results[rng.randrange(len(results))] for _ in range(len(results))]
        samples.append(average_precision(drawn))
    samples.sort()
    low = samples[int(0.025 * (rounds - 1))]
    high = samples[int(0.975 * (rounds - 1))]
    return (low, high)


def image_paths(root: Path) -> list[Path]:
    """Every image under ``root``, in a stable order."""
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def label_path_for(image: Path, images_root: Path, labels_root: Path) -> Path:
    """Mirror the image's subdirectory under ``labels_root`` with a .txt suffix."""
    relative = image.relative_to(images_root)
    return (labels_root / relative).with_suffix(".txt")
