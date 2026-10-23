"""The full cycle against a real PostgreSQL, with the real migrations.

These tests are skipped when no server is reachable. They are the only place
where the model, the migration and the unit of work are exercised together.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from backend.adapters.outbound.db.unit_of_work import SqlAlchemyUnitOfWork
from backend.app.models.detection_event import DetectionEvent
from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.models.vehicle import Vehicle
from backend.application.close_session import close_session
from backend.application.register_observation import register_observation
from backend.domain.events import GateEvent
from backend.domain.vehicle import Plate, VehicleAttributes
from backend.tests.fakes import FakePublisher

pytestmark = pytest.mark.integration

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)
ATTRS = VehicleAttributes(
    vehicle_type="car",
    vehicle_type_confidence=0.9,
    color="plata",
    color_confidence=0.8,
)


class _Clock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


def _event(
    camera_id: str = "gate-1",
    observed_at: datetime = NOW,
    track_id: int = 1,
) -> GateEvent:
    return GateEvent(
        camera_id=camera_id,
        track_id=track_id,
        box=(0, 0, 100, 100),
        observed_at=observed_at,
        attributes=ATTRS,
        detection_confidence=0.91,
    )


def _uow(db):
    return SqlAlchemyUnitOfWork(db)


def test_entry_and_exit_with_the_same_plate_close_the_session(db) -> None:
    with _uow(db) as unit:
        opened = register_observation(
            event=_event(), gate="entry", image_path="/tmp/crop.jpg",
            plate=Plate(text="1234ABC", confidence=0.95),
            model_name="yolo26x", latency_ms=12.5,
            uow=unit, publisher=FakePublisher(), clock=_Clock(NOW),
        )

    with db() as session:
        vehicle = session.get(Vehicle, uuid.UUID(opened.vehicle_id))
        assert vehicle is not None
        assert vehicle.license_plate == "1234ABC"
        assert vehicle.license_plate_confidence == 0.95
        parking = session.get(ParkingSession, uuid.UUID(opened.session_id))
        assert parking is not None
        assert parking.status == SessionStatus.ACTIVE

    exit_at = NOW + timedelta(hours=1)
    with _uow(db) as unit:
        closed = register_observation(
            event=_event("gate-2", exit_at, track_id=2),
            gate="exit", image_path=None,
            plate=Plate(text="1234-ABC", confidence=0.95),
            uow=unit, publisher=FakePublisher(), clock=_Clock(exit_at),
        )

    assert closed.matched_session_id == opened.session_id

    with db() as session:
        parking = session.get(ParkingSession, uuid.UUID(opened.session_id))
        assert parking is not None
        assert parking.status == SessionStatus.COMPLETED
        assert parking.exit_gate == "gate-2"
        assert parking.duration_seconds == 3600
        assert parking.close_reason == "plate_match"
        assert parking.closed_by is None
        assert session.scalar(select(func.count(DetectionEvent.id))) == 2


def test_an_exit_without_a_plate_leaves_the_session_open(db) -> None:
    with _uow(db) as unit:
        opened = register_observation(
            event=_event(), gate="entry", image_path=None,
            uow=unit, publisher=FakePublisher(), clock=_Clock(NOW),
        )

    with _uow(db) as unit:
        unmatched = register_observation(
            event=_event("gate-2", NOW + timedelta(minutes=20), track_id=2),
            gate="exit", image_path=None,
            uow=unit, publisher=FakePublisher(),
            clock=_Clock(NOW + timedelta(minutes=20)),
        )

    assert unmatched.matched_session_id is None
    with db() as session:
        parking = session.get(ParkingSession, uuid.UUID(opened.session_id))
        assert parking is not None
        assert parking.status == SessionStatus.ACTIVE


def test_an_operator_closes_a_session_with_a_reason(db) -> None:
    with _uow(db) as unit:
        opened = register_observation(
            event=_event(), gate="entry", image_path=None,
            uow=unit, publisher=FakePublisher(), clock=_Clock(NOW),
        )

    later = NOW + timedelta(hours=2)
    with _uow(db) as unit:
        outcome = close_session(
            session_id=opened.session_id,
            operator="ana",
            reason="left without a read",
            uow=unit,
            publisher=FakePublisher(),
            clock=_Clock(later),
        )

    assert outcome.duration_seconds == 7200
    with db() as session:
        parking = session.get(ParkingSession, uuid.UUID(opened.session_id))
        assert parking is not None
        assert parking.status == SessionStatus.COMPLETED
        assert parking.exit_time == later
        assert parking.close_reason == "left without a read"
        assert parking.closed_by == "ana"
