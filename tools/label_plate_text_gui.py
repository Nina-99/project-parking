"""Type the plate text for already-verified boxes (end-to-end measurement).

    uv run python tools/label_plate_text_gui.py \
        --verified /home/nina/FNI2025/ProjectParking/plates_det/verified.jsonl \
        --output /home/nina/FNI2025/ProjectParking/plates_det/text.jsonl \
        --proposals /home/nina/FNI2025/ProjectParking/plates_det/proposed.jsonl

Why this exists: the detector was measured on *where* the plate is, and the CRNN
on *what* a tight plate crop says. Neither tells you what the assembled system
does on a real vehicle crop, because that needs the text of a plate the system
has never seen -- and the box review never asked for it.

The text is the ground truth for recognition, so a person writes it: Paddle's
read is wrong often enough that accepting it blindly would train the model on
its own mistakes. But most boxes already carry a proposal, and editing a
prefilled string is far faster than typing from zero. ``--proposals`` prefills
it; the human still has to press Enter, and the saved row records whether the
suggestion was accepted verbatim (``source: paddle``) or corrected
(``source: human``), so the pseudo-label never masquerades as a human label.

A sample is enough. This is not a training set; it is the ruler.

Controls
    A-Z 0-9       type the plate (uppercase; the checkpoint alphabet is 0-9 A-Z)
    Backspace     delete the last character
    Space         clear the whole line (start over on a bad suggestion)
    Enter         save and move on
    Tab           skip this crop (illegible, plate not really there)
    Esc           quit and keep what is saved

Progress is appended, so this can be done in several sittings.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CROP_W = 900
CROP_H = 420
ZOOM_H = 190
GREEN = (80, 220, 80)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
YELLOW = (60, 220, 240)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--proposals",
        type=Path,
        default=None,
        help="proposed.jsonl to prefill Paddle's read. The human still accepts it explicitly.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="How many crops to label this sitting. 0 means every "
        "one still pending; a small default silently truncates "
        "a dataset that looks finished.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--backup-dir", type=Path, default=None, help="Mirror the output here on exit."
    )
    return parser


def done_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["stem"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_proposals(path: Path | None) -> dict[str, str]:
    """Paddle's read by stem, last write winning.

    Only well-formed plate strings survive: a proposal that is empty, has
    spaces, or mixes in characters outside the checkpoint alphabet would
    prefill a line the reviewer has to erase before typing anyway.
    """
    if path is None or not path.exists():
        return {}
    proposals: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text") or "").strip().upper().replace(" ", "")
        if text and text.isascii() and text.isalnum():
            proposals[row["stem"]] = text
    return proposals


def label_source(prefill: str, edited: bool) -> str:
    """Who really produced the label: the model, or the person who confirmed it."""
    return "human" if edited or not prefill else "paddle"


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """Fit into an exact canvas, centred, preserving aspect ratio.

    Returning the scale and the offsets matters: without them the box drawn on
    the fitted image lands somewhere else, and the reviewer reads the wrong
    region.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    offset_x, offset_y = (width - new_w) // 2, (height - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = cv2.resize(
        image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )
    return canvas, scale, offset_x, offset_y


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.verified.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    positives = [r for r in rows if r.get("box")]

    finished = done_stems(args.output)
    queue = [r for r in positives if r["stem"] not in finished]
    random.Random(args.seed).shuffle(queue)
    if args.limit > 0:
        queue = queue[: args.limit]

    proposals = load_proposals(args.proposals)
    suggested = sum(1 for r in queue if r["stem"] in proposals)

    print(
        f"[text] {len(positives)} verified boxes, {len(finished)} typed, "
        f"{len(queue)} to do, {suggested} with a Paddle suggestion",
        flush=True,
    )
    if not queue:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    window = "plate text"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    saved = 0
    unreadable = 0
    for index, row in enumerate(queue):
        image = cv2.imread(row["image"])
        if image is None:
            # Skipping in silence is how a relative path turns into a review
            # that looks finished while a third of the crops were never shown.
            unreadable += 1
            continue
        x1, y1, x2, y2 = row["box"]
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 2, min(x2, width))
        y2 = max(y1 + 2, min(y2, height))

        full, scale, offset_x, offset_y = letterbox(image, CROP_W, CROP_H)
        cv2.rectangle(
            full,
            (offset_x + int(x1 * scale), offset_y + int(y1 * scale)),
            (offset_x + int(x2 * scale), offset_y + int(y2 * scale)),
            GREEN,
            2,
        )

        tight = image[y1:y2, x1:x2]
        zoom, _, _, _ = letterbox(tight, CROP_W, ZOOM_H)

        prefill = proposals.get(row["stem"], "")
        text = prefill
        edited = False
        accepted = False
        while True:
            header = np.zeros((40, CROP_W, 3), dtype=np.uint8)
            state = ""
            if prefill and not edited:
                state = "   PADDLE SUGGESTION: Enter=accept, Space=clear"
            cv2.putText(
                header,
                f"{index + 1}/{len(queue)}   saved={saved}   "
                "letters/digits=type   Backspace=erase   Enter=save   "
                "Tab=skip   Esc=quit" + state,
                (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                GREY,
                1,
                cv2.LINE_AA,
            )

            entry = np.zeros((60, CROP_W, 3), dtype=np.uint8)
            shown = text if text else "_"
            colour = YELLOW if text else GREY
            cv2.putText(
                entry, shown, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA
            )
            cv2.putText(
                entry,
                f"len={len(text)}",
                (CROP_W - 110, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GREY,
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window, np.vstack([header, full, zoom, entry]))
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if key in (13, 10):  # Enter
                accepted = True
                break
            if key == 9:  # Tab = skip
                text = ""
                break
            if key == 27:  # Esc = quit
                cv2.destroyWindow(window)
                if args.backup_dir is not None:
                    args.backup_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(args.output, args.backup_dir / args.output.name)
                print(f"[text] quit early; {saved} saved this session")
                return
            if key in (8, 127):  # Backspace
                text = text[:-1]
                edited = True
                continue
            if key == 32:  # Space clears the suggestion
                text = ""
                edited = True
                continue
            char = chr(key).upper()
            if char.isalnum() and char.isascii():
                text += char
                edited = True

        if accepted and text:
            # Provenance matters: a suggestion accepted verbatim is a Paddle
            # label with a human signature, and the thesis must be able to tell
            # it apart from text typed from scratch.
            source = label_source(prefill, edited)
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "stem": row["stem"],
                            "image": row["image"],
                            "box": row["box"],
                            "text": text,
                            "source": source,
                        }
                    )
                    + "\n"
                )
            saved += 1
        else:
            # Skipped, or accepted with an empty line: record it so the sample
            # does not silently reappear on resume and so the counts add up.
            with args.output.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "stem": row["stem"],
                            "image": row["image"],
                            "box": row["box"],
                            "text": None,
                            "skipped": True,
                        }
                    )
                    + "\n"
                )

    cv2.destroyWindow(window)
    if args.backup_dir is not None:
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output, args.backup_dir / args.output.name)
    if unreadable:
        print(
            f"[text] {unreadable} crops could not be read and were skipped: "
            f"check the paths in {args.verified}"
        )
    print(f"[text] saved {saved} texts this session -> {args.output}")


if __name__ == "__main__":
    main()
