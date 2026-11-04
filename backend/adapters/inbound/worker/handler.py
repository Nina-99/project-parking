# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""One frame in, confirmed gate events out.

This module is the body of the Celery task, separated from Celery so it can be
called in a test with a fake analyzer and a recording unit of work. It knows
nothing about RTSP, JPEG or the broker.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime

from backend.adapters.inbound.worker.dependencies import WorkerDependencies
from backend.adapters.inbound.worker.ports import Frame
from backend.application.register_observation import register_observation
from backend.domain.events import GateDetection
from backend.domain.session import as_utc
from backend.domain.vehicle import VehicleAttributes


def handle_frame(
    dependencies: WorkerDependencies,
    *,
    camera_id: str,
    frame: Frame,
    gate: str,
    captured_at: datetime | None = None,
) -> dict[str, object]:
    """Detect, aggregate and persist the events confirmed by this frame.

    ``captured_at`` is when the frame was sampled, not when the task ran. A
    backlog in the queue must not move a vehicle to a later hour.
    """
    observed_at = as_utc(captured_at) if captured_at is not None else dependencies.clock.now()

    started = time.perf_counter()
    detections = dependencies.analyzer.process(frame)
    latency_ms = (time.perf_counter() - started) * 1000.0
    events = dependencies.aggregator_for(
        camera_id, frame.shape[1], frame.shape[0],
    ).update(
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
                detection_confidence=detection.detection_confidence,
            )
            for detection in detections
        ],
        observed_at,
    )

    outcomes = []
    for event in events:
        image_path = dependencies.crop_store.save(frame, event.box, camera_id)
        plate = None
        if dependencies.plate_reader is not None:
            x1, y1, x2, y2 = event.box
            plate = dependencies.plate_reader.read(frame[y1:y2, x1:x2])
        with dependencies.uow_factory() as uow:
            outcome = register_observation(
                event=event,
                gate=gate,
                image_path=image_path,
                plate=plate,
                uow=uow,
                publisher=dependencies.publisher,
                clock=dependencies.clock,
                model_name=dependencies.model_name,
                latency_ms=latency_ms,
            )
        outcomes.append(asdict(outcome))

    return {
        "camera_id": camera_id,
        "gate": gate,
        "captured_at": observed_at.isoformat(),
        "detections": len(detections),
        "events": outcomes,
    }
