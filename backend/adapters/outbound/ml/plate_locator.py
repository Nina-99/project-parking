# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Plate locator: finds the plate box inside a vehicle crop.

The CRNN reads tight plate crops; the worker hands it vehicle crops. Without a
locator in between, the CRNN is shown a whole car squeezed into 192x48 and reads
nothing. This is the missing stage, trained by ``tools/train_plate_detector.py``
from boxes a human verified.

Kept separate from the reader on purpose: localization and recognition fail for
different reasons, and when OCR accuracy drops the first question is whether the
box was right.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.domain.ports import Frame


class PlateLocator:
    """Single-class YOLO detector wrapped down to ``box | None``."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str | None = None,
        confidence: float = 0.25,
        min_width: int = 24,
        min_height: int = 8,
    ) -> None:
        from ultralytics import YOLO  # heavy: only the worker needs it

        self.device = device
        self.confidence = confidence
        self.min_width = min_width
        self.min_height = min_height
        self.model = YOLO(str(checkpoint))

    def locate(self, vehicle_crop: Frame) -> tuple[int, int, int, int] | None:
        """Highest-confidence plate box in crop coordinates, or ``None``."""
        if vehicle_crop.size == 0:
            return None

        results = self.model.predict(
            source=vehicle_crop,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        best: tuple[int, int, int, int] | None = None
        best_conf = -1.0
        for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
            x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
            if (x2 - x1) < self.min_width or (y2 - y1) < self.min_height:
                continue
            if float(conf) > best_conf:
                best_conf = float(conf)
                best = (x1, y1, x2, y2)
        return best


def crop_plate(vehicle_crop: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Tight crop with a small margin: the CRNN was trained on crops that include
    a sliver of the plate's border, and trimming to the exact glyphs changes the
    aspect ratio it learned."""
    height, width = vehicle_crop.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(1, int((x2 - x1) * 0.04))
    pad_y = max(1, int((y2 - y1) * 0.08))
    return vehicle_crop[max(0, y1 - pad_y):min(height, y2 + pad_y),
                        max(0, x1 - pad_x):min(width, x2 + pad_x)]


def build_locator(checkpoint: Path | None, device: str | None = None) -> Any:
    """``None`` when no detector is configured, so callers can stay optional."""
    if checkpoint is None or not Path(checkpoint).exists():
        return None
    return PlateLocator(Path(checkpoint), device=device)
