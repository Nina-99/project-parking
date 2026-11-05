# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""OCR ablation harness: real-only vs. real+synthetic, scored on real only.

Protocol (this is what backs H1.4 in the thesis): split the real plates by
recording session — never by image — train one OCR model on real-only and one
on real+synthetic, and score both on the held-out REAL test set with character
accuracy, CER and exact match. Synthetic never enters the test set.

This module owns the split and the metrics. Model training plugs in later; the
split files it consumes are written by ``tools/prepare_ocr_ablation.py``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (character level)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[len(b)]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit distance over reference length."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


@dataclass
class OcrScores:
    n: int
    char_accuracy: float
    mean_cer: float
    exact_match: float


def score_pairs(pairs: list[tuple[str, str]]) -> OcrScores:
    """Score (reference, hypothesis) pairs.

    ``char_accuracy`` is 1 - CER averaged per sample, matching the thesis
    criterion (character-level accuracy >= 90%, CER <= 10%).
    """
    if not pairs:
        return OcrScores(n=0, char_accuracy=0.0, mean_cer=0.0, exact_match=0.0)
    cers = [cer(ref, hyp) for ref, hyp in pairs]
    exact = sum(1 for ref, hyp in pairs if ref == hyp) / len(pairs)
    mean_cer = sum(cers) / len(cers)
    return OcrScores(
        n=len(pairs),
        char_accuracy=1.0 - mean_cer,
        mean_cer=mean_cer,
        exact_match=exact,
    )


def split_by_video(
    stems_by_video: dict[str, list[str]],
    test_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[list[str], list[str]]:
    """Split stems into train/test by recording session (video).

    Splitting by video — not by image — keeps every frame of one vehicle on a
    single side. A random per-image split would leak: the same car appears in
    dozens of near-identical frames and would land on both sides at once.
    """
    rng = random.Random(seed)
    videos = sorted(stems_by_video)
    rng.shuffle(videos)
    n_test = max(1, int(round(len(videos) * test_fraction)))
    test_videos = set(videos[:n_test])
    train = [s for v in videos if v not in test_videos for s in stems_by_video[v]]
    test = [s for v in videos if v in test_videos for s in stems_by_video[v]]
    return train, test


def split_by_plate(
    stems_by_plate: dict[str, list[str]],
    test_fraction: float = 0.2,
    seed: int = 7,
    max_per_plate: int = 0,
) -> tuple[list[str], list[str]]:
    """Split stems into train/test so no plate string appears on both sides.

    A session split keeps the same vehicle on one side but still puts the same
    *plate* on both: half the crops in this dataset share their string with
    another recording block. For a recogniser that inflates exact-match, because
    the test then asks the model to reproduce a sequence it already memorised.
    Plate identity is the tighter constraint and the one that matters.

    Groups are ordered by a hash of (seed, plate), not by insertion order, so
    the split is reproducible and stable as crops are added: a new crop of an
    existing plate joins its plate's side, and never reshuffles the rest.

    ``max_per_plate`` caps how many crops of one plate reach the test. Without
    it a single car that crossed the gate thirty times accounts for a tenth of
    the test set, and the exact-match becomes a statement about that car. The
    crops left out are dropped, not moved to train: the same plate must never
    sit on both sides.
    """
    ordered = sorted(
        stems_by_plate,
        key=lambda plate: hashlib.sha256(f"{seed}:{plate}".encode()).hexdigest(),
    )

    def kept(plate: str) -> list[str]:
        group = sorted(stems_by_plate[plate])
        return group[:max_per_plate] if max_per_plate else group

    total = sum(len(kept(plate)) for plate in ordered)
    target = test_fraction * total
    test_plates: set[str] = set()
    counted = 0
    for plate in ordered:
        if counted + len(kept(plate)) <= target:
            test_plates.add(plate)
            counted += len(kept(plate))
    train = [s for p_ in ordered if p_ not in test_plates for s in stems_by_plate[p_]]
    test = [s for p_ in ordered if p_ in test_plates for s in kept(p_)]
    return train, test


def collect_real_plates_by_text(root: Path) -> dict[str, list[str]]:
    """Map plate text -> stems for every clean real crop (synth excluded)."""
    base = Path(root) / "plates"
    img_dir = base / "images"
    lbl_dir = base / "labels"
    by_plate: dict[str, list[str]] = {}
    if not img_dir.exists():
        return by_plate
    for image in sorted(img_dir.glob("**/*.jpg")):
        if "synth" in image.relative_to(img_dir).parts:
            continue
        label = lbl_dir / image.relative_to(img_dir).with_suffix(".txt")
        if not label.exists():
            continue
        text = label.read_text(encoding="utf-8").strip()
        if text:
            by_plate.setdefault(text, []).append(image.stem)
    return by_plate


def collect_real_plates(root: Path) -> dict[str, list[str]]:
    """Map video -> stems for the clean real plates (excludes synth)."""
    base = Path(root) / "plates"
    img_dir = base / "images"
    lbl_dir = base / "labels"
    by_video: dict[str, list[str]] = {}
    if not img_dir.exists():
        return by_video
    for video_dir in sorted(img_dir.iterdir()):
        if not video_dir.is_dir() or video_dir.name == "synth":
            continue
        stems = [
            p.stem
            for p in sorted(video_dir.glob("*.jpg"))
            if (lbl_dir / video_dir.name / f"{p.stem}.txt").exists()
        ]
        if stems:
            by_video[video_dir.name] = stems
    return by_video
