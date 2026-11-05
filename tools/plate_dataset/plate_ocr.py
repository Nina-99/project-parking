# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""License-plate pre-labelling with PaddleOCR.

PaddleOCR is used as a bootstrap *pseudo-labeller*, not as ground truth: its
detections become review candidates that a human corrects before the samples
enter the clean dataset. The engine is imported lazily because loading the
Paddle weights is expensive and only needed when a plate is actually read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]

# Real plates are short alphanumeric strings that always mix digits and letters
# (brand lettering like "TOYOTA" is all letters and must be rejected).
_PLATE_TEXT = re.compile(r"^[A-Z0-9]{4,9}$")
_MIN_RATIO = 1.8
_MAX_RATIO = 4.5


def _looks_like_plate_text(text: str) -> bool:
    return bool(_PLATE_TEXT.match(text)) and any(ch.isdigit() for ch in text)


@dataclass
class PlateRead:
    """A single plate candidate found inside a vehicle crop."""

    bbox: Box          # x1, y1, x2, y2 in vehicle-crop coordinates
    text: str
    confidence: float
    sharpness: float


def plate_sharpness(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian over the tight plate crop."""
    if image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_plate_like(box: Box, crop_w: int, crop_h: int, y_min: float, y_max: float,
                   min_px: int) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < min_px or h < max(8, min_px // 4):
        return False
    ratio = w / float(h)
    if not (_MIN_RATIO <= ratio <= _MAX_RATIO):
        return False
    center_y = (y1 + y2) / 2.0 / float(crop_h)
    return y_min <= center_y <= y_max


class PlateReader:
    """Thin wrapper around PaddleOCR tuned for plate-like detections."""

    def __init__(self, min_conf: float = 0.60, min_plate_px: int = 40,
                 plate_y_min: float = 0.35, plate_y_max: float = 1.00,
                 search_y_min: float = 0.40, upscale: float = 2.0,
                 lang: str = "latin", use_gpu: bool = False) -> None:
        self.min_conf = min_conf
        self.min_plate_px = min_plate_px
        self.plate_y_min = plate_y_min
        self.plate_y_max = plate_y_max
        # Plates live low on a vehicle and are small; searching only the bottom
        # strip and upscaling it focuses the detector and helps recognition.
        self.search_y_min = search_y_min
        self.upscale = upscale
        self.lang = lang
        self.use_gpu = use_gpu
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR  # heavy import, done lazily

            try:
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang,
                                         use_gpu=self.use_gpu, show_log=False)
            except TypeError:
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang)
        return self._engine

    @staticmethod
    def _center_in_boxes(box: Box, boxes: list[Box]) -> bool:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in boxes)

    def read(self, vehicle_crop: np.ndarray,
             exclude_boxes: list[Box] | None = None) -> PlateRead | None:
        """Return the most plausible plate inside ``vehicle_crop`` or ``None``.

        ``exclude_boxes`` are crop-local pixel boxes (e.g. where the recorder's
        on-screen display was mapped) whose detections must be ignored.
        """
        if vehicle_crop.size == 0:
            return None
        exclude_boxes = exclude_boxes or []

        crop_h, crop_w = vehicle_crop.shape[:2]
        strip_y0 = int(crop_h * self.search_y_min)
        strip = vehicle_crop[strip_y0:crop_h, :]
        if strip.size == 0:
            strip = vehicle_crop
            strip_y0 = 0

        scale = self.upscale
        if scale and scale != 1.0:
            strip = cv2.resize(strip, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        try:
            result = self._ensure_engine().ocr(strip, cls=True)
        except Exception:
            return None

        if not result or result[0] is None:
            return None

        best: PlateRead | None = None
        for line in result[0]:
            raw_box, (text, conf) = line[0], line[1]
            xs = [p[0] for p in raw_box]
            ys = [p[1] for p in raw_box]
            # Map the detection from the upscaled strip back to vehicle-crop coords.
            box = (
                int(min(xs) / scale),
                int(min(ys) / scale) + strip_y0,
                int(max(xs) / scale),
                int(max(ys) / scale) + strip_y0,
            )

            if not _is_plate_like(box, crop_w, crop_h, self.plate_y_min,
                                  self.plate_y_max, self.min_plate_px):
                continue
            if self._center_in_boxes(box, exclude_boxes):
                continue

            x1, y1, x2, y2 = box
            tight = vehicle_crop[max(0, y1):y2, max(0, x1):x2]
            cleaned = str(text).strip().upper().replace(" ", "").replace("-", "")
            if not _looks_like_plate_text(cleaned):
                continue

            read = PlateRead(
                bbox=box,
                text=cleaned,
                confidence=float(conf),
                sharpness=plate_sharpness(tight),
            )
            if best is None or read.confidence > best.confidence:
                best = read

        return best
