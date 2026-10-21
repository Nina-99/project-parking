from __future__ import annotations

from celery import shared_task


@shared_task
def sample_parking_task() -> str:
    """Sample Celery task placeholder for the parking app."""
    return "Parking task executed"
