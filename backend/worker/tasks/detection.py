"""Celery task for gate inference.

This is an inbound adapter and nothing else: decode the frame, ask the analyzer
for the vehicles in it, let the domain decide what counts as an event, and hand
each event to the write use case. SQL, matching and publishing live below.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from celery import shared_task

from backend.adapters.inbound.worker.tracking import aggregator_for
from backend.adapters.outbound.bus.redis_publisher import RedisEventPublisher
from backend.adapters.outbound.db.unit_of_work import SqlAlchemyUnitOfWork
from backend.adapters.outbound.time import SystemClock
from backend.app.core.config import settings
from backend.application.register_observation import register_observation
from backend.domain.events import Box, GateDetection
from backend.domain.vehicle import VehicleAttributes
from backend.worker.db import get_session_factory


def save_crop(frame: np.ndarray, box: Box, camera_id: str) -> Path | None:
    """Persist the vehicle crop so the incident search has a thumbnail."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    directory = Path(settings.ENTRY_IMAGE_DIR) / camera_id
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = SystemClock().now().strftime("%Y%m%dT%H%M%S")
        path = directory / f"{stamp}_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(str(path), crop)
    except OSError:
        return None  # a missing thumbnail must not lose the row
    return path


@shared_task(bind=True, name="detection.ping")
def ping_task(self) -> str:  # noqa: ARG001
    """Simple ping to verify Celery is working."""
    return "pong from detection worker"


@shared_task(bind=True, name="detection.process_frame")
def process_frame(self, camera_id: str, frame_bytes: bytes,  # noqa: ARG001
                  gate: str = "entry") -> dict[str, object]:
    """Detect the vehicles in one frame and persist the confirmed events.

    ``gate`` is ``entry`` or ``exit``. A frame can confirm zero, one or several
    tracks; each confirmed track is persisted exactly once, when the aggregator
    says it is real. That is what keeps one car from opening several sessions.
    """
    from backend.worker.engine import DetectionEngine

    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"camera_id": camera_id, "status": "undecodable"}

    detections = DetectionEngine.get_instance().process(frame)
    events = aggregator_for(camera_id).update(
        camera_id,
        [
            GateDetection(
                box=detection.box,
                attributes=VehicleAttributes(
                    vehicle_type=detection.vehicle_type,
                    vehicle_type_confidence=detection.vehicle_type_confidence,
                    color=detection.color,
                    color_confidence=detection.color_confidence,
                ),
            )
            for detection in detections
        ],
        SystemClock().now(),
    )

    outcomes = []
    for event in events:
        crop_path = save_crop(frame, event.box, camera_id)
        with SqlAlchemyUnitOfWork(get_session_factory()) as uow:
            outcome = register_observation(
                event=event,
                gate=gate,
                image_path=str(crop_path) if crop_path else None,
                uow=uow,
                publisher=RedisEventPublisher(settings.CELERY_BROKER_URL),
                clock=SystemClock(),
            )
        outcomes.append(asdict(outcome))

    return {
        "camera_id": camera_id,
        "gate": gate,
        "detections": len(detections),
        "events": outcomes,
    }
