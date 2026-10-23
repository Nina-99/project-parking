"""What the worker handler needs from its collaborators.

These are ports of an inbound adapter, not of the domain: they describe the
things around a frame (a model, a crop store, a clock) so the task can be
exercised with fakes and no GPU.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from backend.domain.events import Box

Frame = NDArray[np.uint8]


class DetectionLike(Protocol):
    """The attributes the handler reads from a model detection."""

    box: Box
    vehicle_type: str | None
    vehicle_type_confidence: float | None
    color: str | None
    color_confidence: float | None


class VehicleAnalyzer(Protocol):
    def process(self, frame: Frame) -> Sequence[DetectionLike]: ...


class CropStore(Protocol):
    def save(self, frame: Frame, box: Box, camera_id: str) -> str | None: ...
