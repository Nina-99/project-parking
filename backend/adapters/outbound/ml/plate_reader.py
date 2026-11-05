# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Choose the plate reader from configuration.

Off by default: until one reader is measured on the frozen test set, auto-close
would be a guess dressed as a fact. Both adapters live behind the same port, so
the comparison is a config change, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path

from backend.app.core.config import settings
from backend.domain.ports import PlateReader


def build_plate_reader() -> PlateReader | None:
    if not settings.OCR_ENABLED:
        return None

    if settings.OCR_READER == "crnn":
        from backend.adapters.outbound.ml.plate_crnn import CrnnPlateReader
        from backend.adapters.outbound.ml.plate_locator import build_locator

        device = "cuda" if settings.OCR_USE_GPU else "cpu"
        return CrnnPlateReader(
            Path(settings.OCR_CRNN_CHECKPOINT),
            device=device,
            min_confidence=settings.OCR_MIN_CONFIDENCE,
            # Without a locator the CRNN receives a vehicle crop and reads
            # garbage: it was trained on tight plate crops.
            locator=build_locator(
                Path(settings.PLATE_DETECTOR_PATH) if settings.PLATE_DETECTOR_PATH else None,
                device=device,
            ),
        )

    from backend.adapters.outbound.ml.plate_paddle import PaddlePlateReader

    return PaddlePlateReader(
        min_conf=settings.OCR_MIN_CONFIDENCE,
        use_gpu=settings.OCR_USE_GPU,
    )
