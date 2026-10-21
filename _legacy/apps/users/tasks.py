from __future__ import annotations

from celery import shared_task


@shared_task
def sample_user_task() -> str:
    """Sample Celery task placeholder for the users app."""
    return "User task executed"
