# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Shared test doubles for the application layer.

They record what the use case asks for, in order, so a test can assert the
contract (and the commit-before-publish ordering) without a database.
"""
from __future__ import annotations

from datetime import UTC, datetime

from backend.domain.matching import OpenSession
from backend.domain.vehicle import Plate, VehicleAttributes

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


class FakeVehicles:
    def __init__(self) -> None:
        self.added: list[tuple[VehicleAttributes, Plate | None, str | None]] = []

    def add(
        self,
        attributes: VehicleAttributes,
        plate: Plate | None,
        image_path: str | None,
    ) -> str:
        self.added.append((attributes, plate, image_path))
        return "vehicle-1"


class FakeSessions:
    def __init__(self, active: list[OpenSession] | None = None) -> None:
        self._active = active or []
        self.opened: list[tuple[str, str, datetime]] = []
        self.closed: list[dict] = []
        self.duration_seconds = 3600
        self.failure: Exception | None = None

    def open(self, vehicle_id: str, gate: str, at: datetime) -> str:
        self.opened.append((vehicle_id, gate, at))
        return "session-1"

    def active(self) -> list[OpenSession]:
        return self._active

    def close(
        self,
        session_id: str,
        at: datetime,
        *,
        gate: str | None = None,
        reason: str | None = None,
        operator: str | None = None,
    ) -> int:
        if self.failure is not None:
            raise self.failure
        self.closed.append({
            "session_id": session_id,
            "at": at,
            "gate": gate,
            "reason": reason,
            "operator": operator,
        })
        return self.duration_seconds


class FakeDetections:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, **fields: object) -> None:
        self.records.append(fields)


class FakeUnitOfWork:
    def __init__(self, active: list[OpenSession] | None = None) -> None:
        self.vehicles = FakeVehicles()
        self.sessions = FakeSessions(active)
        self.detections = FakeDetections()
        self.commits = 0
        self.log: list[str] = []

    def __enter__(self) -> FakeUnitOfWork:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def commit(self) -> None:
        self.commits += 1
        self.log.append("commit")


class FakePublisher:
    def __init__(self, log: list[str] | None = None) -> None:
        self.payloads: list[dict] = []
        self._log = log if log is not None else []

    def publish(self, payload) -> bool:
        self.payloads.append(dict(payload))
        self._log.append("publish")
        return True


class FixedClock:
    def now(self) -> datetime:
        return NOW


def open_session(plate: str | None = None, confidence: float = 0.9) -> OpenSession:
    return OpenSession(
        session_id="session-old",
        vehicle_id="vehicle-old",
        entry_time=NOW,
        license_plate=plate,
        plate_confidence=confidence,
    )
