# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The single write path: one confirmed gate event becomes one vehicle.

The Celery worker ends up here, so there is one answer to "what happens when a
vehicle is seen". An entry opens a session; an exit closes one only when an
identifier proves it, and is otherwise recorded as unmatched. The inference row
is written in the same transaction: telemetry that can outlive its vehicle is
worse than no telemetry. Publishing happens after the commit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.domain.events import GateEvent
from backend.domain.matching import ExitVehicle, find_match
from backend.domain.ports import Clock, EventPublisher, UnitOfWork
from backend.domain.vehicle import Plate

GATES = ("entry", "exit")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationOutcome:
    vehicle_id: str
    session_id: str | None
    matched_session_id: str | None
    event_type: str
    published: bool


def register_observation(
    *,
    event: GateEvent,
    gate: str,
    image_path: str | None,
    uow: UnitOfWork,
    publisher: EventPublisher,
    clock: Clock,
    plate: Plate | None = None,
    model_name: str | None = None,
    latency_ms: float | None = None,
) -> ObservationOutcome:
    """Persist one gate event and announce it.

    ``plate`` is the OCR result when there is one. Without it an exit cannot be
    matched, and the session stays open: a wrong match would empty the lot
    silently, an unmatched exit is visible.

    ``latency_ms`` is the inference time for this frame, measured by the worker.
    It is optional because a test or a replay may not have a model at all.
    """
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")

    # The crossing decides the direction when it was seen. The camera's role is
    # only a fallback: at a gate used both ways, a camera fixed to "exit" would
    # record an entry as an exit.
    direction = event.direction or gate

    vehicle_id = uow.vehicles.add(event.attributes, plate, image_path)

    session_id: str | None = None
    matched_session_id: str | None = None
    if direction == "entry":
        session_id = uow.sessions.open(vehicle_id, event.camera_id, event.observed_at)
    else:
        match = find_match(
            ExitVehicle(
                license_plate=plate.text if plate else None,
                plate_confidence=plate.confidence if plate else None,
            ),
            uow.sessions.active(),
        )
        if match is not None:
            uow.sessions.close(
                match.session_id,
                event.observed_at,
                gate=event.camera_id,
                reason="plate_match",
            )
            matched_session_id = match.session_id

    uow.detections.record(
        session_id=session_id or matched_session_id,
        camera_id=event.camera_id,
        observed_at=event.observed_at,
        detection_confidence=event.detection_confidence,
        box=event.box,
        crop_image_path=image_path,
        model_name=model_name or "unknown",
        inference_latency_ms=latency_ms,
        track_id=event.track_id,
        direction=direction,
    )
    uow.commit()

    if direction == "entry":
        event_type = "vehicle_entry"
    elif matched_session_id is not None:
        event_type = "vehicle_exit"
    else:
        event_type = "vehicle_exit_unmatched"

    published = publisher.publish({
        "type": event_type,
        "camera_id": event.camera_id,
        "vehicle_id": vehicle_id,
        "session_id": session_id or matched_session_id,
        "vehicle_type": event.attributes.vehicle_type,
        "color": event.attributes.color,
        "timestamp": clock.now().isoformat(),
    })

    logger.info(
        "gate_event type=%s camera=%s track=%s vehicle=%s session=%s plate=%s latency_ms=%s",
        event_type,
        event.camera_id,
        event.track_id,
        vehicle_id,
        session_id or matched_session_id,
        bool(plate),
        None if latency_ms is None else round(latency_ms, 1),
    )

    return ObservationOutcome(
        vehicle_id=vehicle_id,
        session_id=session_id,
        matched_session_id=matched_session_id,
        event_type=event_type,
        published=published,
    )
