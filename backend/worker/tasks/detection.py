"""Celery tasks for YOLO inference and vehicle attribute classification.

The capture side samples a frame every half second and dispatches it here. One
frame yields at most one vehicle of interest -- the largest detection, which is
the one at the gate -- and its attributes are written to ``Vehicle`` so the
incident search and the statistics have something to read.

What is deliberately not here yet: matching an exit to its entry. That needs a
plate or a ReID vector, and until it exists an exit is recorded as a vehicle
without closing anybody's session. Writing a guess would corrupt the occupancy
count, which is the one number the operator trusts.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from celery import shared_task
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.matching import ExitVehicle, OpenSession, find_match
from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.models.vehicle import Vehicle
from backend.worker.db import sync_session

EVENTS_CHANNEL = "parking:events"


def pick_vehicle(detections: list) -> object | None:
    """The vehicle at a gate is the largest one in the frame."""
    return detections[0] if detections else None


def vehicle_fields(detection: object) -> dict:
    """Columns for a ``Vehicle`` row, straight from a detection."""
    return {
        "vehicle_type": detection.vehicle_type,
        "vehicle_type_confidence": detection.vehicle_type_confidence,
        "color": detection.color,
        "color_confidence": detection.color_confidence,
    }


def save_crop(frame: np.ndarray, detection: object, camera_id: str) -> Path | None:
    """Persist the vehicle crop so the incident search has a thumbnail."""
    x1, y1, x2, y2 = detection.box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    directory = Path(settings.ENTRY_IMAGE_DIR) / camera_id
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = directory / f"{stamp}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(path), crop)
    except OSError:
        return None  # a missing thumbnail must not lose the row
    return path


def open_sessions(session: Session) -> list[OpenSession]:
    """Every active session, with the identifier its vehicle carries."""
    rows = session.execute(
        select(ParkingSession, Vehicle)
        .join(Vehicle, ParkingSession.vehicle_id == Vehicle.id)
        .where(ParkingSession.status == SessionStatus.ACTIVE)
    ).all()
    return [
        OpenSession(
            session_id=str(parking.id),
            vehicle_id=str(vehicle.id),
            entry_time=parking.entry_time,
            license_plate=vehicle.license_plate,
            plate_confidence=vehicle.license_plate_confidence,
        )
        for parking, vehicle in rows
    ]


def close_matching_session(
    session: Session, exit_vehicle: ExitVehicle, camera_id: str,
) -> str | None:
    """Close the session this exit belongs to, or leave everything untouched.

    Only an exact identifier closes a session. Anything else -- no plate, a plate
    below the confidence floor, a plate nobody is inside with -- returns None and
    the session stays open, so the occupancy count drifts upward visibly instead
    of silently closing somebody else's session.
    """
    match = find_match(exit_vehicle, open_sessions(session))
    if match is None:
        return None
    parking = session.get(ParkingSession, match.session_id)
    if parking is None:
        return None
    parking.exit_gate = camera_id
    parking.exit_time = datetime.now(UTC)
    elapsed = (parking.exit_time - parking.entry_time).total_seconds()
    parking.duration_seconds = max(0, int(elapsed))
    parking.status = SessionStatus.COMPLETED
    return str(parking.id)


def publish_event(payload: dict) -> bool:
    """Announce the event on Redis; a broker outage must not undo the write."""
    try:
        import redis

        client = redis.Redis.from_url(settings.CELERY_BROKER_URL)
        client.publish(EVENTS_CHANNEL, json.dumps(payload))
        client.close()
        return True
    except Exception:
        return False


@shared_task(bind=True, name="detection.ping")
def ping_task(self) -> str:  # noqa: ARG001
    """Simple ping to verify Celery is working."""
    return "pong from detection worker"


@shared_task(bind=True, name="detection.process_frame")
def process_frame(self, camera_id: str, frame_bytes: bytes,  # noqa: ARG001
                  gate: str = "entry") -> dict:
    """Detect the vehicle in a frame and record it.

    ``gate`` is ``entry`` or ``exit``. Only an entry opens a parking session;
    an exit is recorded and left unmatched until plate or ReID matching exists.
    """
    from backend.worker.engine import DetectionEngine

    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"camera_id": camera_id, "status": "undecodable"}

    detections = DetectionEngine.get_instance().process(frame)
    vehicle = pick_vehicle(detections)
    if vehicle is None:
        return {"camera_id": camera_id, "detections": 0}

    crop_path = save_crop(frame, vehicle, camera_id)

    with sync_session() as session:
        row = Vehicle(**vehicle_fields(vehicle), entry_image_path=str(crop_path) if crop_path else None)
        session.add(row)
        session.flush()
        vehicle_id = str(row.id)

        session_id: str | None = None
        matched_session_id: str | None = None
        if gate == "entry":
            parking = ParkingSession(
                vehicle_id=row.id,
                entry_gate=camera_id,
                status=SessionStatus.ACTIVE,
            )
            session.add(parking)
            session.flush()
            session_id = str(parking.id)
        else:
            # The engine has no OCR yet, so this is always None today and the
            # branch is honest about it: an exit that cannot be proven leaves the
            # session open. Wiring it now means only the plate extraction has to
            # change when OCR arrives.
            matched_session_id = close_matching_session(
                session, ExitVehicle(license_plate=None, plate_confidence=None), camera_id)

    event_type = "vehicle_entry" if gate == "entry" else "vehicle_exit"
    if gate == "exit" and matched_session_id is None:
        event_type = "vehicle_exit_unmatched"

    published = publish_event({
        "type": event_type,
        "camera_id": camera_id,
        "vehicle_id": vehicle_id,
        "session_id": session_id or matched_session_id,
        "vehicle_type": vehicle.vehicle_type,
        "color": vehicle.color,
        "timestamp": datetime.now(UTC).isoformat(),
    })

    return {
        "camera_id": camera_id,
        "gate": gate,
        "detections": len(detections),
        "vehicle_id": vehicle_id,
        "session_id": session_id,
        "matched_session_id": matched_session_id,
        "event": event_type,
        "vehicle_type": vehicle.vehicle_type,
        "color": vehicle.color,
        "published": published,
    }
