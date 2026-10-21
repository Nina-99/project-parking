from __future__ import annotations

from celery import shared_task


@shared_task
def sample_vehicle_task() -> str:
    """Sample Celery task placeholder for the vehicles app."""
    return "Vehicle task executed"
