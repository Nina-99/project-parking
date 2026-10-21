"""
Celery tasks for YOLO inference, OCR, and vehicle classification.

These run on the Celery worker (GPU-enabled container) and process
images asynchronously.
"""
from __future__ import annotations

from celery import shared_task
from backend.app.core.config import settings


@shared_task(bind=True, name="detection.ping")
def ping_task(self) -> str:
    """Simple ping to verify Celery is working."""
    return f"pong from detection worker"


@shared_task(bind=True, name="detection.process_entry")
def process_entry(self, session_id: str, image_bytes: bytes) -> dict:
    """
    Process an entry gate image asynchronously.

    1. Load YOLO model (cached)
    2. Detect vehicle
    3. Crop license plate region
    4. Run OCR on plate
    5. Classify vehicle color
    6. Classify vehicle type/model
    7. Update DB with results
    8. Broadcast via WebSocket
    """
    # TODO: Implement when YOLO model is trained
    # from backend.detection.engine import DetectionEngine
    # engine = DetectionEngine.get_instance()
    # result = engine.process_entry_image(image_bytes)
    # ... update DB via SQLAlchemy ...
    # ... broadcast via Redis pub/sub ...

    return {
        "session_id": session_id,
        "status": "processed",
        "note": "Detection engine not yet implemented — placeholder",
    }


@shared_task(bind=True, name="detection.process_exit")
def process_exit(self, session_id: str, image_bytes: bytes) -> dict:
    """
    Process an exit gate image.

    1. Load YOLO model
    2. Detect vehicle
    3. Match with entry vehicle via ReID (or license plate)
    4. Close the session
    5. Broadcast exit event
    """
    return {
        "session_id": session_id,
        "status": "processed",
        "note": "Exit processing not yet implemented — placeholder",
    }
