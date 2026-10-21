"""
Celery application configuration.

Workers se ejecutan en un proceso separado (o container)
y procesan las tareas de inferencia YOLO de forma asíncrona.
"""
from __future__ import annotations

from celery import Celery

from backend.app.core.config import settings

celery_app = Celery(
    "parking",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["backend.worker.tasks.detection"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 min max per task
    task_soft_time_limit=240,
)
