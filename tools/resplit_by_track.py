"""Rebuild a YOLO dataset split by vehicle track, eliminating frame leakage.

    uv run python tools/resplit_by_track.py \
        --dataset "/home/nina/FNI2025/ProjectParking/dataset/type car v1" \
        --output  "/home/nina/FNI2025/ProjectParking/dataset/type car grouped"

Roboflow splits randomly by image, so a few frames of the same vehicle land in
both train and valid — near-duplicate leakage that inflates validation scores.
This tool groups every image by its ``(video, track_id)`` and assigns a whole
track to exactly one split, stratified by the track's dominant class so each
class still appears in every split. Deterministic via ``--seed``.

Output keeps the YOLO layout (images + labels per split) and a ``data.yaml``, so
``tools/train_type_classifier.py`` consumes it unchanged.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")

# Roboflow appends "_jpg.rf.<hash>" and may add an augmentation copy per image.
# Two naming conventions coexist in the exports:
#   C.OCCIDENTE_..._track_867_rank1_frame_48693   (track id, optional underscore)
#   1017_rank1_frame_50634                        (numeric prefix, no track)
# Grouping keys must cover both, or frames of the same vehicle leak across splits.
_TRACK_RE = re.compile(r"^(?P<video>.+?)_track_?(?P<track>\d+)")
_NUM_RE = re.compile(r"^(?P<id>\d+)_")


def track_key(stem: str) -> str:
    """Stable vehicle key for a crop name, covering both naming conventions."""
    match = _TRACK_RE.match(stem)
    if match is not None:
        return f"{match.group('video')}_track{match.group('track')}"
    match = _NUM_RE.match(stem)
    if match is not None:
        return f"num_{match.group('id')}"
    return stem


def read_class(label_path: Path) -> int | None:
    """Class of the largest box in a YOLO label file (the main vehicle)."""
    best_class: int | None = None
    best_area = -1.0
    if not label_path.exists():
        return None
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            area = float(parts[3]) * float(parts[4])
        except ValueError:
            continue
        if area > best_area:
            best_area = area
            best_class = cls
    return best_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-split a YOLO dataset by track.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    src = args.dataset
    names = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))["names"]

    # Gather every image with its track and dominant class.
    items: list[tuple[Path, Path, str, int]] = []
    for split in SPLITS:
        for img in sorted((src / split / "images").glob("*.jpg")):
            label = src / split / "labels" / f"{img.stem}.txt"
            cls = read_class(label)
            if cls is None:
                continue
            track = track_key(img.stem)
            items.append((img, label, track, cls))
    print(f"[resplit] images: {len(items)}")

    # Dominant class per track decides its stratum.
    track_classes: dict[str, Counter] = defaultdict(Counter)
    for _, _, track, cls in items:
        track_classes[track][cls] += 1
    primary = {t: c.most_common(1)[0][0] for t, c in track_classes.items()}
    print(f"[resplit] unique tracks: {len(primary)}")

    strata: dict[int, list[str]] = defaultdict(list)
    for track, cls in primary.items():
        strata[cls].append(track)

    rng = random.Random(args.seed)
    assign: dict[str, str] = {}
    for cls in sorted(strata):
        tracks = sorted(strata[cls])
        rng.shuffle(tracks)
        n = len(tracks)
        n_test = round(n * args.test_frac)
        n_val = round(n * args.val_frac)
        for t in tracks[:n_test]:
            assign[t] = "test"
        for t in tracks[n_test:n_test + n_val]:
            assign[t] = "valid"
        for t in tracks[n_test + n_val:]:
            assign[t] = "train"

    # Copy images + labels into the new split folders.
    for split in SPLITS:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for img, label, track, cls in items:
        split = assign[track]
        counts[split][names[cls]] += 1
        shutil.copy2(img, args.output / split / "images" / img.name)
        shutil.copy2(label, args.output / split / "labels" / label.name)

    yaml_text = (
        f"train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(names)}\nnames: {names}\n"
    )
    (args.output / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("[resplit] per-split class counts (by image):")
    for split in SPLITS:
        row = "  ".join(f"{names[i]}={counts[split][names[i]]}" for i in range(len(names)))
        print(f"  {split:6s}: {row}")
    print(f"[resplit] written to {args.output}")


if __name__ == "__main__":
    main()
