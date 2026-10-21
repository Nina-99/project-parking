from __future__ import annotations

from celery import shared_task


@shared_task
def sample_detection_task() -> str:
    """Sample Celery task placeholder for the detection app."""
    return "Detection task executed"
