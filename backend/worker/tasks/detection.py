"""Celery task for gate inference.

The task is a thin inbound adapter: decode the frame, parse the capture time and
call the handler. Everything that can be tested lives in `handler.py`.
"""
from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
from celery import shared_task

from backend.adapters.inbound.worker.dependencies import build_worker_dependencies
from backend.adapters.inbound.worker.handler import handle_frame


def parse_captured_at(value: str | None) -> datetime | None:
    """ISO instant from the capture process; None when absent or malformed."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@shared_task(bind=True, name="detection.ping")
def ping_task(self) -> str:  # noqa: ARG001
    """Simple ping to verify Celery is working."""
    return "pong from detection worker"


@shared_task(bind=True, name="detection.process_frame")
def process_frame(
    self,  # noqa: ARG001
    camera_id: str,
    frame_bytes: bytes,
    gate: str = "entry",
    captured_at: str | None = None,
) -> dict[str, object]:
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"camera_id": camera_id, "status": "undecodable"}

    return handle_frame(
        build_worker_dependencies(),
        camera_id=camera_id,
        frame=frame,
        gate=gate,
        captured_at=parse_captured_at(captured_at),
    )
