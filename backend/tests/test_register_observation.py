"""The write use case: what a confirmed gate event does to the database.

No database here. The stores are recorders, so the test can assert the order
that matters: the commit happens before anything is published.
"""
from __future__ import annotations

from backend.application.register_observation import register_observation
from backend.domain.events import GateEvent
from backend.domain.vehicle import Plate, VehicleAttributes
from backend.tests.fakes import (
    NOW,
    FakePublisher,
    FakeUnitOfWork,
    FixedClock,
    open_session,
)

ATTRS = VehicleAttributes(vehicle_type="car", vehicle_type_confidence=0.9,
                          color="plata", color_confidence=0.8)


def event(camera_id: str = "gate-1") -> GateEvent:
    return GateEvent(
        camera_id=camera_id,
        track_id=7,
        box=(0, 0, 100, 100),
        observed_at=NOW,
        attributes=ATTRS,
        detection_confidence=0.91,
    )


def test_an_entry_records_the_vehicle_and_opens_a_session() -> None:
    uow = FakeUnitOfWork()

    outcome = register_observation(
        event=event(), gate="entry", image_path="/tmp/crop.jpg",
        uow=uow, publisher=FakePublisher(), clock=FixedClock(),
    )

    assert uow.vehicles.added == [(ATTRS, None, "/tmp/crop.jpg")]
    assert uow.sessions.opened == [("vehicle-1", "gate-1", NOW)]
    assert outcome.session_id == "session-1"
    assert outcome.event_type == "vehicle_entry"


def test_an_entry_persists_the_plate_it_was_given() -> None:
    uow = FakeUnitOfWork()
    plate = Plate(text="1234-ABC", confidence=0.94)

    register_observation(
        event=event(), gate="entry", image_path=None, plate=plate,
        uow=uow, publisher=FakePublisher(), clock=FixedClock(),
    )

    _, stored_plate, _ = uow.vehicles.added[0]
    assert stored_plate == plate, "an unread plate cannot close a session later"


def test_the_event_is_published_only_after_the_commit() -> None:
    uow = FakeUnitOfWork()

    register_observation(
        event=event(), gate="entry", image_path=None,
        uow=uow, publisher=FakePublisher(uow.log), clock=FixedClock(),
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

    assert uow.sessions.closed == [{
        "session_id": "session-old",
        "at": NOW,
        "gate": "gate-2",
        "reason": "plate_match",
        "operator": None,
    }]
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


def test_the_event_keeps_its_inference_metadata() -> None:
    uow = FakeUnitOfWork()

    register_observation(
        event=event(), gate="entry", image_path="/tmp/crop.jpg",
        uow=uow, publisher=FakePublisher(), clock=FixedClock(),
        model_name="yolo26x", latency_ms=42.5,
    )

    record = uow.detections.records[0]
    assert record["session_id"] == "session-1"
    assert record["camera_id"] == "gate-1"
    assert record["detection_confidence"] == 0.91
    assert record["model_name"] == "yolo26x"
    assert record["inference_latency_ms"] == 42.5
    assert record["track_id"] == 7
    assert record["crop_image_path"] == "/tmp/crop.jpg"
