# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Pseudo-label plate bounding boxes inside vehicle crops with PaddleOCR.

    uv run python tools/pseudo_label_plates.py \
        --images /home/nina/FNI2025/ProjectParking/datasets/pool_20261021/images \
        --output /home/nina/FNI2025/ProjectParking/plates_det/proposed.jsonl

This is step 1 of three for the plate detector:

1. ``pseudo_label_plates.py`` (this file) proposes a box per vehicle crop.
2. ``label_plate_box_gui.py`` lets a human accept, correct or delete it.
3. ``build_plate_detection_dataset.py`` exports the verified boxes as YOLO.

PaddleOCR is a *pseudo*-labeller, exactly as the thesis describes for the plate
dataset: nothing here is training data until a person has looked at the box.

Rows are written as JSONL so an interrupted run resumes instead of restarting::

    {"stem": "1017_rank1_frame_50634_b00", "box": [x1, y1, x2, y2],
     "text": "2688YXS", "conf": 0.93, "sharpness": 143.2}

``box`` is null when PaddleOCR found no plate-like candidate. Those rows are not
failures to discard -- they are the *hard negatives* a detector needs so it does
not hallucinate a plate on every car.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.plate_ocr import PlateReader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True,
                        help="Directory of vehicle crops.")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSONL to write (one row per crop).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N new crops (0 = all).")
    parser.add_argument("--min-conf", type=float, default=0.30,
                        help="Keep candidates at or above this OCR confidence. "
                             "Low on purpose: the human review is the filter, and "
                             "a rejected box is still useful as a hard negative.")
    parser.add_argument("--lang", default="latin")
    parser.add_argument("--shuffle-seed", type=int, default=7,
                        help="Deterministic order, so two runs propose the same "
                             "crops in the same sequence and a partial review "
                             "stays usable.")
    return parser


def already_done(output: Path) -> set[str]:
    done: set[str] = set()
    if not output.exists():
        return done
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["stem"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    images = sorted(args.images.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no .jpg under {args.images}")

    # Stable order: sorting by a hash of the name keeps the sequence identical
    # across runs without needing to store it.
    import hashlib

    def order(path: Path) -> bytes:
        return hashlib.blake2b(path.stem.encode(), digest_size=8,
                               key=str(args.shuffle_seed).encode()).digest()

    images.sort(key=order)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.output)
    pending = [p for p in images if p.stem not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"[pseudo] {len(images)} crops, {len(done)} already labelled, "
          f"{len(pending)} to do", flush=True)
    if not pending:
        return

    reader = PlateReader(min_conf=args.min_conf, lang=args.lang)

    t0 = time.perf_counter()
    found = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for index, path in enumerate(pending, 1):
            crop = cv2.imread(str(path))
            row: dict = {"stem": path.stem, "image": str(path), "box": None,
                         "text": None, "conf": None, "sharpness": None}
            if crop is not None:
                read = reader.read(crop)
                if read is not None:
                    found += 1
                    row.update(box=[int(v) for v in read.bbox], text=read.text,
                               conf=round(read.confidence, 4),
                               sharpness=round(read.sharpness, 2))
            handle.write(json.dumps(row) + "\n")
            if index % 100 == 0:
                handle.flush()
                elapsed = time.perf_counter() - t0
                rate = index / elapsed
                remaining = (len(pending) - index) / rate if rate else 0
                print(f"[pseudo] {index}/{len(pending)}  {rate:.2f} crops/s  "
                      f"~{remaining / 60:.0f} min left  found={found}", flush=True)

    total = len(done) + len(pending)
    print(f"[pseudo] done: {total} rows, {found} proposals in this run "
          f"({found / len(pending) * 100:.1f}%) -> {args.output}")


if __name__ == "__main__":
    main()
