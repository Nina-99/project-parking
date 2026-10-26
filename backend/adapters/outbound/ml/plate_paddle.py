# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""PaddleOCR plate reader.

Locates the plate inside a vehicle crop and reads it. This is the operational
baseline: it already exists as the dataset's pseudo-labeller and works without a
custom detector.

The logic mirrors ``tools/plate_dataset/plate_ocr.py``. It is duplicated while
the training run owns that file; unify both after the run ends. The non-obvious
rules (search only the bottom strip, upscale, reject non-plate aspect ratios)
are the reason this is not a one-line call to PaddleOCR.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from backend.domain.matching import looks_like_plate, normalize_plate
from backend.domain.ports import Frame
from backend.domain.vehicle import Plate

_MIN_RATIO = 1.8
_MAX_RATIO = 4.5


def is_plate_candidate(
    box: tuple[int, int, int, int],
    crop_height: int,
    y_min: float,
    y_max: float,
    min_px: int,
) -> bool:
    """Shape and position rules a plate-like detection must satisfy."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width < min_px or height < max(8, min_px // 4):
        return False
    ratio = width / float(height)
    if not (_MIN_RATIO <= ratio <= _MAX_RATIO):
        return False
    center_y = (y1 + y2) / 2.0 / float(crop_height)
    return y_min <= center_y <= y_max


class PaddlePlateReader:
    def __init__(
        self,
        min_conf: float = 0.60,
        min_plate_px: int = 40,
        plate_y_min: float = 0.35,
        plate_y_max: float = 1.00,
        search_y_min: float = 0.40,
        upscale: float = 2.0,
        lang: str = "latin",
        use_gpu: bool = False,
    ) -> None:
        self.min_conf = min_conf
        self.min_plate_px = min_plate_px
        self.plate_y_min = plate_y_min
        self.plate_y_max = plate_y_max
        self.search_y_min = search_y_min
        self.upscale = upscale
        self.lang = lang
        self.use_gpu = use_gpu
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            from paddleocr import PaddleOCR  # heavy import, done on first read

            try:
                self._engine = PaddleOCR(
                    use_angle_cls=True, lang=self.lang,
                    use_gpu=self.use_gpu, show_log=False,
                )
            except TypeError:  # PaddleOCR 3.x dropped those kwargs
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang)
        return self._engine

    def read(self, vehicle_crop: Frame) -> Plate | None:
        if vehicle_crop.size == 0:
            return None

        crop_h = vehicle_crop.shape[0]
        strip_y0 = int(crop_h * self.search_y_min)
        strip = vehicle_crop[strip_y0:crop_h, :]
        if strip.size == 0:
            strip = vehicle_crop
            strip_y0 = 0

        scale = self.upscale
        if scale and scale != 1.0:
            strip = np.asarray(
                cv2.resize(strip, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC),
                dtype=np.uint8,
            )

        try:
            result = self._ensure_engine().ocr(strip, cls=True)
        except Exception:
            return None
        if not result or result[0] is None:
            return None

        best: Plate | None = None
        for line in result[0]:
            raw_box, (text, confidence) = line[0], line[1]
            xs = [point[0] for point in raw_box]
            ys = [point[1] for point in raw_box]
            box = (
                int(min(xs) / scale),
                int(min(ys) / scale) + strip_y0,
                int(max(xs) / scale),
                int(max(ys) / scale) + strip_y0,
            )
            if not is_plate_candidate(
                box, crop_h, self.plate_y_min, self.plate_y_max,
                self.min_plate_px,
            ):
                continue
            if float(confidence) < self.min_conf:
                continue
            cleaned = normalize_plate(str(text))
            if not looks_like_plate(cleaned):
                continue
            candidate = Plate(text=cleaned, confidence=float(confidence))
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best
