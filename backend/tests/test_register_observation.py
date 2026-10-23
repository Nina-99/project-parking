"""The write use case: what a confirmed gate event does to the database.

No database here. The stores are recorders, so the test can assert the order
that matters: the commit happens before anything is published.
"""
from __future__ import annotations

from datetime import UTC, datetime

from backend.application.register_observation import register_observation
from backend.domain.events import GateEvent
from backend.domain.matching import OpenSession
from backend.domain.vehicle import Plate, VehicleAttributes

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)
ATTRS = VehicleAttributes(vehicle_type="car", vehicle_type_confidence=0.9,
                          color="plata", color_confidence=0.8)


def event(camera_id: str = "gate-1") -> GateEvent:
    return GateEvent(
        camera_id=camera_id,
        track_id=7,
        box=(0, 0, 100, 100),
        observed_at=NOW,
        attributes=ATTRS,
    )


class FakeVehicles:
    def __init__(self) -> None:
        self.added: list[tuple[VehicleAttributes, str | None]] = []

    def add(self, attributes: VehicleAttributes, image_path: str | None) -> str:
        self.added.append((attributes, image_path))
        return "vehicle-1"


class FakeSessions:
    def __init__(self, active: list[OpenSession] | None = None) -> None:
        self._active = active or []
        self.opened: list[tuple[str, str, datetime]] = []
        self.closed: list[tuple[str, datetime, str]] = []

    def open(self, vehicle_id: str, gate: str, at: datetime) -> str:
        self.opened.append((vehicle_id, gate, at))
        return "session-1"

    def active(self) -> list[OpenSession]:
        return self._active

    def close(self, session_id: str, at: datetime, gate: str) -> None:
        self.closed.append((session_id, at, gate))


class FakeUnitOfWork:
    def __init__(self, active: list[OpenSession] | None = None) -> None:
        self.vehicles = FakeVehicles()
        self.sessions = FakeSessions(active)
        self.commits = 0
        self.log: list[str] = []

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


def test_an_entry_records_the_vehicle_and_opens_a_session() -> None:
    uow = FakeUnitOfWork()
    publisher = FakePublisher()

    outcome = register_observation(
        event=event(), gate="entry", image_path="/tmp/crop.jpg",
        uow=uow, publisher=publisher, clock=FixedClock(),
    )

    assert uow.vehicles.added == [(ATTRS, "/tmp/crop.jpg")]
    assert uow.sessions.opened == [("vehicle-1", "gate-1", NOW)]
    assert outcome.session_id == "session-1"
    assert outcome.event_type == "vehicle_entry"


def test_the_event_is_published_only_after_the_commit() -> None:
    uow = FakeUnitOfWork()
    publisher = FakePublisher(uow.log)

    register_observation(
        event=event(), gate="entry", image_path=None,
        uow=uow, publisher=publisher, clock=FixedClock(),
    )

    assert uow.log == ["commit", "publish"]


def test_an_exit_without_a_plate_leaves_the_session_open() -> None:
    uow = FakeUnitOfWork(active=[open_session(plate="ABC123")])

    outcome = register_observation(
        event=event("gate-2"), gate="exit", image_path=None,
        uow=uow, publisher=FakePublisher(), clock=FixedClock(),
    )

    assert uow.sessions.closed == []
    assert outcome.matched_session_id is None
    assert outcome.event_type == "vehicle_exit_unmatched"


def test_an_exit_with_a_matching_plate_closes_that_session() -> None:
    uow = FakeUnitOfWork(active=[open_session(plate="ABC-123")])

    outcome = register_observation(
        event=event("gate-2"), gate="exit", image_path=None,
        uow=uow, publisher=FakePublisher(), clock=FixedClock(),
        plate=Plate(text="abc 123", confidence=0.95),
    )

    assert uow.sessions.closed == [("session-old", NOW, "gate-2")]
    assert outcome.event_type == "vehicle_exit"


def test_a_low_confidence_plate_is_not_evidence() -> None:
    uow = FakeUnitOfWork(active=[open_session(plate="ABC123")])

    outcome = register_observation(
        event=event("gate-2"), gate="exit", image_path=None,
        uow=uow, publisher=FakePublisher(), clock=FixedClock(),
        plate=Plate(text="ABC123", confidence=0.2),
    )

    assert uow.sessions.closed == []
    assert outcome.event_type == "vehicle_exit_unmatched"


def test_a_gate_that_is_neither_entry_nor_exit_is_rejected() -> None:
    try:
        register_observation(
            event=event(), gate="side", image_path=None,
            uow=FakeUnitOfWork(), publisher=FakePublisher(), clock=FixedClock(),
        )
    except ValueError as error:
        assert "gate must be one of" in str(error)
    else:
        raise AssertionError("an unknown gate must not be persisted")
