# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Review vehicle boxes for H1.1. A person accepts, deletes or draws.

    uv run python tools/label_boxes_gui.py \
        --root /home/nina/Smart_Parking_dataset/detection --blind-every 4

The detector proposes, the person decides. A proposal is a starting point and
nothing else: the label that gets scored is the one a human left on screen. A
deleted proposal is not wasted -- it becomes a false positive the detector is
penalised for, which is exactly the number H1.1 needs.

``--blind-every 4`` opens one image in four with no proposals at all and marks
the banner red. That subset is the only one that can measure a *miss*: with a
box already drawn, the reviewer stops looking for vehicles the detector did not
propose. The acceptance rate is written to ``review_journal.jsonl`` and reported
by ``--check``, so the thesis can say how much of the labelling the model did.

Controls
    click inside a box   select it (red)
    d / Delete           delete the selected box
    drag on empty space  draw a new box
    a                    accept every proposal on screen
    u                    undo: restore the proposals as they arrived
    n                    no vehicles here -> save an empty label (a negative)
    Enter / Space        save and advance
    s                    skip: leave the image unlabelled for another session
    q / Esc              quit; the image on screen is not saved

Progress is written per image, so a crash costs at most the image on screen.
Images whose label file already exists are skipped unless ``--force`` is given.

Why one class: H1.1 asks whether a vehicle is found. The body type is H1.2 and
is measured on crops, where the answer is not ambiguous.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection_dataset import (  # noqa: E402
    Box,
    image_paths,
    label_path_for,
    write_labels,
)

MAX_VIEW_W = 1100
MAX_VIEW_H = 720

PROPOSAL = (80, 220, 80)
SELECTED = (60, 60, 240)
DRAWN = (0, 200, 255)
TEXT = (255, 255, 255)


class Viewport:
    """Maps between frame pixels and the window, so the mouse means what it looks like."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.scale = min(MAX_VIEW_W / width, MAX_VIEW_H / height, 1.0)
        self.view_w = max(1, int(round(width * self.scale)))
        self.view_h = max(1, int(round(height * self.scale)))

    def to_view(self, box: Box) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (int(x1 * self.scale), int(y1 * self.scale),
                int(x2 * self.scale), int(y2 * self.scale))

    def to_original(self, x: int, y: int) -> tuple[int, int]:
        return (int(round(x / self.scale)), int(round(y / self.scale)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Dataset root with images/ and proposals.jsonl.")
    parser.add_argument("--force", action="store_true",
                        help="Review images that already have a label file.")
    parser.add_argument("--blind-every", type=int, default=0, metavar="N",
                        help="Review one image in N with no proposals at all. The "
                             "blind subset is the only one that can measure a miss: "
                             "with proposals on screen the annotator stops looking "
                             "for vehicles the detector did not propose. 0 disables.")
    parser.add_argument("--check", action="store_true",
                        help="Do not open a window: report how much is left to label.")
    parser.add_argument("--class-index", type=int, default=0)
    return parser


def is_blind(name: str, every: int) -> bool:
    """Deterministic one-in-N selection, stable across sessions and file order.

    Keyed on the file name rather than its position, so adding footage later
    does not reshuffle which images were blind.
    """
    if every <= 1:
        return False
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    return int(digest, 16) % every == 0


def was_modified(saved: list[Box], proposals: list[Box]) -> bool:
    """Did the reviewer change what arrived on screen?"""
    if len(saved) != len(proposals):
        return True
    for saved_box, proposed in zip(saved, proposals, strict=True):
        if any(abs(a - b) > 2 for a, b in zip(saved_box, proposed, strict=True)):
            return True
    return False


def load_proposals(root: Path) -> dict[str, list[Box]]:
    """Proposals per image, keyed by the path relative to ``images/``."""
    path = root / "proposals.jsonl"
    proposals: dict[str, list[Box]] = {}
    if not path.exists():
        return proposals
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        boxes = [tuple(int(v) for v in box) for box in row.get("boxes", [])]
        # ``image`` is already relative to images/: "<camera>/<file>.jpg".
        proposals[row["image"]] = boxes  # type: ignore[assignment]
    return proposals


def check(root: Path) -> None:
    images_root = root / "images"
    labels_root = root / "labels"
    images = image_paths(images_root)
    labelled = [p for p in images
                if label_path_for(p, images_root, labels_root).exists()]
    missing = len(images) - len(labelled)
    empties = 0
    for image in labelled:
        label = label_path_for(image, images_root, labels_root)
        if not label.read_text(encoding="utf-8").strip():
            empties += 1
    print(f"images={len(images)} labelled={len(labelled)} missing={missing} "
          f"negatives={empties}")
    if images:
        print(f"progress={100 * len(labelled) / len(images):.1f}%")

    journal = root / "review_journal.jsonl"
    if journal.exists():
        rows = [json.loads(line) for line in
                journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        blind = sum(1 for row in rows if row.get("blind"))
        untouched = sum(1 for row in rows if not row.get("modified"))
        print(f"reviewed={len(rows)} blind={blind} "
              f"accepted_unmodified={untouched}")
        if rows:
            print(f"acceptance_rate={100 * untouched / len(rows):.1f}%")


def review_image(frame: np.ndarray, proposals: list[Box],
                 blind: bool) -> tuple[str, list[Box]]:
    """One image. Returns ('save'|'skip'|'quit', boxes)."""
    height, width = frame.shape[:2]
    viewport = Viewport(width, height)

    boxes: list[Box] = list(proposals)
    selected = -1
    drag_start: tuple[int, int] | None = None
    drag_now: tuple[int, int] | None = None
    outcome = "skip"

    window = "label vehicle boxes"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal selected, drag_start, drag_now
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = -1
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = viewport.to_view(box)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hit = index
            if hit >= 0:
                selected = hit
                drag_start = drag_now = None
            else:
                selected = -1
                drag_start = (x, y)
                drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            if drag_now is not None:
                x1, y1 = viewport.to_original(*drag_start)
                x2, y2 = viewport.to_original(*drag_now)
                if abs(x2 - x1) >= 8 and abs(y2 - y1) >= 8:
                    boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            drag_start = drag_now = None

    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = cv2.resize(frame, (viewport.view_w, viewport.view_h),
                            interpolation=cv2.INTER_AREA)
        for index, box in enumerate(boxes):
            colour = SELECTED if index == selected else (
                PROPOSAL if index < len(proposals) else DRAWN)
            cv2.rectangle(canvas, viewport.to_view(box)[:2], viewport.to_view(box)[2:],
                          colour, 2)
        if drag_start and drag_now:
            cv2.rectangle(canvas, drag_start, drag_now, DRAWN, 1)

        banner = (f"{'BLIND' if blind else f'proposals {len(proposals)}'} | "
                  f"{len(boxes)} boxes | click=select d=delete drag=draw "
                  f"a=accept-all u=undo n=none enter=save s=skip q=quit")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26),
                      (30, 30, 150) if blind else (30, 30, 30), -1)
        cv2.putText(canvas, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32):
            outcome = "save"
            break
        if key in (ord("q"), 27):
            outcome = "quit"
            break
        if key == ord("s"):
            outcome = "skip"
            break
        if key == ord("n"):
            boxes = []
            selected = -1
        elif key == ord("a"):
            for box in proposals:
                if box not in boxes:
                    boxes.append(box)
        elif key == ord("u"):
            boxes = list(proposals)
            selected = -1
        elif key in (ord("d"), 8, 127) and 0 <= selected < len(boxes):
            boxes.pop(selected)
            selected = -1

    cv2.destroyWindow(window)
    return outcome, boxes


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    images_root = args.root / "images"
    labels_root = args.root / "labels"

    if args.check:
        check(args.root)
        return

    proposals = load_proposals(args.root)
    images = image_paths(images_root)
    if not images:
        raise SystemExit(f"no images under {images_root}")

    pending = [p for p in images
               if args.force or not label_path_for(p, images_root, labels_root).exists()]
    blind_count = sum(1 for p in pending
                      if is_blind(str(p.relative_to(images_root)), args.blind_every))
    print(f"[label] {len(pending)} of {len(images)} images to review "
          f"({blind_count} blind)")

    journal = (args.root / "review_journal.jsonl").open("a", encoding="utf-8")
    for index, image in enumerate(pending, start=1):
        frame = cv2.imread(str(image))
        if frame is None:
            print(f"[label] cannot read {image}", flush=True)
            continue
        key = str(image.relative_to(images_root))
        blind = is_blind(key, args.blind_every)
        shown = [] if blind else proposals.get(key, [])
        outcome, boxes = review_image(frame, shown, blind)
        label = label_path_for(image, images_root, labels_root)
        if outcome == "save":
            height, width = frame.shape[:2]
            write_labels(label, boxes, width, height, args.class_index)
            journal.write(json.dumps({
                "image": key,
                "blind": blind,
                "proposals": len(shown),
                "boxes": len(boxes),
                "modified": was_modified(boxes, shown),
                "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }) + "\n")
            journal.flush()
        elif outcome == "quit":
            print(f"[label] stopped at {index - 1} of {len(pending)}", flush=True)
            break
        print(f"[label] {index}/{len(pending)} {key} -> {outcome} "
              f"({len(boxes)} boxes{' , blind' if blind else ''})", flush=True)

    journal.close()
    check(args.root)


if __name__ == "__main__":
    main()
