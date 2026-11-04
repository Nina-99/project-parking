# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The ports the application needs, and nothing else.

Each protocol here exists because something below it needs to be replaced in a
test or lives in a different process. If a new protocol cannot answer "what
invariant does this protect?", it does not belong in this file.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from backend.domain.events import Box
from backend.domain.matching import OpenSession
from backend.domain.vehicle import Plate, VehicleAttributes

Frame = NDArray[np.uint8]


class PlateReader(Protocol):
    """Read a plate from a vehicle crop, or say it could not.

    The crop is a vehicle, not a plate: locating the plate is the adapter's job.
    A reader that is not sure returns None; a guess would close the wrong
    session later.
    """

    def read(self, vehicle_crop: Frame) -> Plate | None: ...


class VehicleStore(Protocol):
    def add(
        self,
        attributes: VehicleAttributes,
        plate: Plate | None,
        image_path: str | None,
    ) -> str: ...


class SessionStore(Protocol):
    def open(self, vehicle_id: str, gate: str, at: datetime) -> str: ...

    def active(self) -> list[OpenSession]: ...

    def close(
        self,
        session_id: str,
        at: datetime,
        *,
        gate: str | None = None,
        reason: str | None = None,
        operator: str | None = None,
    ) -> int: ...


class EventPublisher(Protocol):
    def publish(self, payload: Mapping[str, object]) -> bool: ...


class DetectionStore(Protocol):
    """One row per confirmed event, for model health and latency."""

    def record(
        self,
        *,
        session_id: str | None,
        camera_id: str,
        observed_at: datetime,
        detection_confidence: float,
        box: Box,
        crop_image_path: str | None,
        model_name: str,
        inference_latency_ms: float | None,
        track_id: int,
        direction: str | None = None,
    ) -> None: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class UnitOfWork(Protocol):
    """One transaction, with the stores that share it."""

    vehicles: VehicleStore
    sessions: SessionStore
    detections: DetectionStore

    def commit(self) -> None: ...
