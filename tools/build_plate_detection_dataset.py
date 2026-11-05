# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Export human-verified plate boxes as a YOLO detection dataset.

    uv run python tools/build_plate_detection_dataset.py \
        --verified /home/nina/FNI2025/ProjectParking/plates_det/verified.jsonl \
        --output /home/nina/FNI2025/ProjectParking/plates_det/dataset

Step 3 of three. One class (``plate``). Crops whose review said "no plate" become
images with an *empty* label file: YOLO treats those as background, and without
them the detector learns to put a plate on every car.

The split is a stable hash of the recording session, never of the frame. Two
crops of the same vehicle (``_b00``, ``_b01``) must not land on opposite sides of
the split, or the test set is measuring memorisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ("train", "valid", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--force", action="store_true",
                        help="Wipe the output directory first.")
    return parser


def session_of(stem: str) -> str:
    """Recording block, independent of the vehicle track or the crop index.

    ``C.OCCIDENTE_ch16_main_20260427090012_20260427100012_track1866_f0048936_b00``
    -> ``C.OCCIDENTE_ch16_main_20260427090012_20260427100012``

    Deliberately *not* the track. Splitting by track puts two vehicles from the
    same hour, camera and lighting on opposite sides of the split, so the test
    set measures the conditions the model already saw. Splitting by recording
    block asks the question that matters: does it still find plates on an hour
    it never trained on?
    """
    for marker in ("_track", "_rank"):
        if marker in stem:
            return stem.split(marker)[0]
    return stem


def split_for(session: str, test_cut: int, valid_cut: int) -> str:
    value = int.from_bytes(
        hashlib.blake2b(session.encode("utf-8"), digest_size=4).digest(), "big"
    ) % 100
    if value < test_cut:
        return "test"
    if value < valid_cut:
        return "valid"
    return "train"


def load_verified(path: Path) -> list[dict]:
    """Last review per stem wins, and skipped crops are dropped."""
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[row["stem"]] = row
    return [r for r in latest.values() if r.get("source") != "skipped"]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = load_verified(args.verified)
    if not rows:
        raise SystemExit(f"no usable rows in {args.verified}")

    if args.output.exists():
        if not args.force:
            raise SystemExit(
                f"{args.output} already exists; pass --force to rebuild "
                "(splits are stable, so this is only needed to change the source)"
            )
        shutil.rmtree(args.output)

    for split in SPLITS:
        (args.output / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

    test_cut = round(args.test_frac * 100)
    valid_cut = test_cut + round(args.val_frac * 100)

    written: Counter = Counter()
    positives: Counter = Counter()
    sessions: dict[str, set[str]] = {s: set() for s in SPLITS}

    for row in rows:
        image = Path(row["image"])
        if not image.exists():
            continue
        session = session_of(row["stem"])
        split = split_for(session, test_cut, valid_cut)
        sessions[split].add(session)

        shutil.copy2(image, args.output / split / "images" / image.name)
        label = args.output / split / "labels" / f"{image.stem}.txt"
        box = row.get("box")
        if box:
            import cv2

            crop = cv2.imread(str(image))
            if crop is None:
                continue
            height, width = crop.shape[:2]
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            label.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
            positives[split] += 1
        else:
            label.write_text("", encoding="utf-8")   # explicit background
        written[split] += 1

    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\nnames: ['plate']\n",
        encoding="utf-8",
    )

    total = sum(written.values())
    print(f"[det-dataset] {total} crops from {len(rows)} reviewed rows")
    for split in SPLITS:
        n = written[split]
        pos = positives[split]
        share = pos / n * 100 if n else 0.0
        print(f"  {split:5s}: {n:4d} crops ({pos} with plate, {share:.0f}%)  "
              f"{len(sessions[split])} sessions")
    overlap = (sessions['train'] & sessions['test']) | (sessions['train'] & sessions['valid'])
    print(f"[det-dataset] sessions shared between train and eval: {len(overlap)}")
    print(f"[det-dataset] written to {args.output}")


if __name__ == "__main__":
    main()
