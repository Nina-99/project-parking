# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Interactive body-type and colour labeller.

Presents each pending crop from the attribute queue and records two labels:
body type and predominant colour. Two keystrokes per sample keeps the loop fast
enough to annotate a few thousand crops in one sitting, which is the real cost
of refusing to pseudo-label these attributes.

The window layout mirrors the zone calibrator: the crop on top, a legend bar
underneath, keyboard-driven.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from plate_dataset.attributes import (
    COCO_VEHICLE_NAMES,
    append_discard,
    append_label,
    image_path,
    key_for_index,
    key_to_index,
    pending_samples,
)

WINDOW = "attribute labelling"
_PREVIEW_LONG_EDGE = 760
_BAR_ROW = 26
_BAR_PAD = 14


def _fit(image: np.ndarray) -> np.ndarray:
    """Scale the crop so its longest edge is readable, up or down."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest == 0:
        return image
    scale = _PREVIEW_LONG_EDGE / float(longest)
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=interp)


class AttributeLabeller:
    """Drives the annotation loop over the pending attribute queue."""

    def __init__(self, root: Path, types: list[str], colors: list[str],
                 labels_path: Path | None = None,
                 manifest_path: Path | None = None) -> None:
        self.root = Path(root)
        self.types = types
        self.colors = colors
        self.labels_path = labels_path or (self.root / "attributes" / "labels.jsonl")
        self.manifest_path = manifest_path or (self.root / "attributes" / "manifest.jsonl")

    # -- queue ----------------------------------------------------------------
    def pending(self) -> list[dict]:
        return pending_samples(self.manifest_path, self.labels_path)

    # -- rendering ------------------------------------------------------------
    def _legend(self, position: int, total: int, sample: dict,
                chosen_type: str | None) -> list[str]:
        types = "  ".join(f"{key_for_index(i)}:{name}" for i, name in enumerate(self.types))
        colors = "  ".join(f"{key_for_index(i)}:{name}" for i, name in enumerate(self.colors))
        coco = COCO_VEHICLE_NAMES.get(sample.get("vehicle_class"))
        hint = f"   detector={coco}" if coco else ""
        step = "TYPE" if chosen_type is None else f"COLOUR (type={chosen_type})"
        return [
            f"[{position}/{total}]  {step}   kind={sample.get('kind', '?')}{hint}",
            f"type:  {types}",
            f"color: {colors}",
            "s: skip   d: discard (unusable)   b: back   x / ESC: save & quit",
        ]

    def _compose(self, image: np.ndarray, lines: list[str]) -> np.ndarray:
        canvas = _fit(image)
        needed = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                     for line in lines) + 2 * _BAR_PAD
        width = max(canvas.shape[1], needed)
        if width > canvas.shape[1]:
            pad = np.zeros((canvas.shape[0], width - canvas.shape[1], 3), dtype=np.uint8)
            canvas = np.hstack([canvas, pad])
        bar = np.zeros((_BAR_ROW * len(lines) + _BAR_PAD, width, 3), dtype=np.uint8)
        for n, line in enumerate(lines):
            cv2.putText(bar, line, (_BAR_PAD, _BAR_PAD + _BAR_ROW * n + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([canvas, bar])

    # -- loop -----------------------------------------------------------------
    def run(self) -> int:
        """Annotate pending samples until the queue drains or the user quits.

        Returns the number of labels written.
        """
        pending = self.pending()
        if not pending:
            print("[label] nothing pending — the queue is fully labelled")
            return 0

        print(f"[label] {len(pending)} crop(s) pending; vocabulary: "
              f"{len(self.types)} type(s), {len(self.colors)} colour(s)")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        written = 0
        discarded = 0
        index = 0
        chosen_type: str | None = None

        while 0 <= index < len(pending):
            sample = pending[index]
            crop = cv2.imread(str(image_path(self.root, sample)))
            if crop is None:
                print(f"[warn] crop missing, skipping {sample.get('stem')}")
                index += 1
                chosen_type = None
                continue

            cv2.imshow(WINDOW, self._compose(
                crop, self._legend(index + 1, len(pending), sample, chosen_type)))
            key = cv2.waitKey(0)
            if key in (27, ord("x")):
                break
            char = chr(key) if 0 <= key < 0x110000 else ""
            if char == "s":
                index += 1
                chosen_type = None
                continue
            if char == "b":
                index = max(0, index - 1)
                chosen_type = None
                continue
            if char == "d":
                # Unjudgeable crop: record the discard so it leaves the queue.
                append_discard(self.labels_path, str(sample.get("stem")))
                discarded += 1
                chosen_type = None
                index += 1
                continue

            if chosen_type is None:
                position = key_to_index(char, len(self.types))
                if position is None:
                    continue
                chosen_type = self.types[position]
                continue

            position = key_to_index(char, len(self.colors))
            if position is None:
                continue
            append_label(self.labels_path, str(sample.get("stem")), chosen_type,
                         self.colors[position], self.types, self.colors)
            written += 1
            chosen_type = None
            index += 1

        cv2.destroyAllWindows()
        print(f"[label] {written} label(s) and {discarded} discard(s) written to {self.labels_path}")
        return written
