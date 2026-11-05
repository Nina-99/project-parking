# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Shared test doubles for the application layer.

They record what the use case asks for, in order, so a test can assert the
contract (and the commit-before-publish ordering) without a database.
"""
from __future__ import annotations

from datetime import UTC, datetime

from backend.domain.errors import VehicleNotFoundError
from backend.domain.matching import OpenSession, normalize_plate
from backend.domain.vehicle import Plate, VehicleAttributes

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


class FakeVehicles:
    def __init__(self) -> None:
        self.added: list[tuple[VehicleAttributes, Plate | None, str | None]] = []
        self.plates: dict[str, str | None] = {}
        self.corrected: list[tuple[str, str]] = []
        self.missing: set[str] = set()
        self._next = 0

    def add(
        self,
        attributes: VehicleAttributes,
        plate: Plate | None,
        image_path: str | None,
    ) -> str:
        self.added.append((attributes, plate, image_path))
        self._next += 1
        return f"vehicle-{self._next}"

    def correct_plate(self, vehicle_id: str, plate: str) -> str | None:
        if vehicle_id in self.missing:
            raise VehicleNotFoundError(vehicle_id)
        self.corrected.append((vehicle_id, plate))
        previous = self.plates.get(vehicle_id)
        self.plates[vehicle_id] = plate
        return previous

    def plate_for(self, vehicle_id: str) -> str | None:
        if vehicle_id in self.missing:
            raise VehicleNotFoundError(vehicle_id)
        return self.plates.get(vehicle_id)


class FakeAuthorizations:
    def __init__(self, authorized: set[str] | None = None) -> None:
        self.authorized = {normalize_plate(item) for item in (authorized or set())}
        self.checks: list[str | None] = []

    def is_authorized(self, plate: str | None) -> bool | None:
        self.checks.append(plate)
        if not plate:
            return None
        return normalize_plate(plate) in self.authorized


class FakeValidations:
    def __init__(self) -> None:
        self.added: list[dict] = []

    def add(self, **fields: object) -> None:
        self.added.append(fields)


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
    def __init__(
        self,
        active: list[OpenSession] | None = None,
        authorized: set[str] | None = None,
    ) -> None:
        self.vehicles = FakeVehicles()
        self.sessions = FakeSessions(active)
        self.detections = FakeDetections()
        self.validations = FakeValidations()
        self.authorizations = FakeAuthorizations(authorized)
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
