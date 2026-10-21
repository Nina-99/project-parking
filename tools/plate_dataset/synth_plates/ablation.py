"""OCR ablation harness: real-only vs. real+synthetic, scored on real only.

Protocol (this is what backs H1.4 in the thesis): split the real plates by
recording session — never by image — train one OCR model on real-only and one
on real+synthetic, and score both on the held-out REAL test set with character
accuracy, CER and exact match. Synthetic never enters the test set.

This module owns the split and the metrics. Model training plugs in later; the
split files it consumes are written by ``tools/prepare_ocr_ablation.py``.
"""

from __future__ import annotations

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
            p.stem for p in sorted(video_dir.glob("*.jpg"))
            if (lbl_dir / video_dir.name / f"{p.stem}.txt").exists()
        ]
        if stems:
            by_video[video_dir.name] = stems
    return by_video
