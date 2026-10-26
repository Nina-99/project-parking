# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Cheap motion gate used by the sparse first pass.

The dense second pass must never be fed sparse frames: trackers such as
ByteTrack/BoT-SORT assume frame-to-frame continuity. Instead, a nearly-free
frame-difference signal decides *when* there is something worth processing.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    """Fraction of pixels that changed significantly since the last sample."""

    def __init__(self, downscale: float = 0.25, pixel_threshold: int = 25) -> None:
        self.downscale = downscale
        self.pixel_threshold = pixel_threshold
        self._prev: np.ndarray | None = None

    def energy(self, frame_bgr: np.ndarray) -> float:
        small = cv2.resize(frame_bgr, None, fx=self.downscale, fy=self.downscale,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev is None:
            self._prev = gray
            return 0.0

        diff = cv2.absdiff(self._prev, gray)
        self._prev = gray
        changed = np.count_nonzero(diff > self.pixel_threshold)
        return float(changed) / float(diff.size)

    def reset(self) -> None:
        self._prev = None


def find_active_segments(
    energies: list[float],
    stride: int,
    threshold: float,
    pad_frames: int,
    merge_gap: int,
) -> list[tuple[int, int]]:
    """Turn per-sample motion energies into merged frame windows [start, end).

    ``energies[i]`` is the motion measured between the i-th and (i+1)-th sampled
    frame. Each window is padded by ``pad_frames`` on both sides so a vehicle
    that starts entering just before a sample is not clipped.
    """
    active = [i for i, e in enumerate(energies) if e >= threshold]
    if not active:
        return []

    runs: list[list[int]] = [[active[0], active[0]]]
    for i in active[1:]:
        if (i - runs[-1][1]) * stride <= merge_gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])

    windows: list[list[int]] = []
    for start, end in runs:
        a = max(0, start * stride - pad_frames)
        b = end * stride + stride + pad_frames
        if windows and a - windows[-1][1] <= merge_gap:
            windows[-1][1] = b
        else:
            windows.append([a, b])

    return [(a, b) for a, b in windows]
