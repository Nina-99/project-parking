# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Celery application for the detection worker."""
from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

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
    # Publishing must fail fast. Defaults retry until the broker answers, and a
    # producer on the capture path measured 107 seconds blocked before giving up
    # when Redis was down -- during which the gate camera stopped sampling. An
    # outage should cost one frame, not two minutes of footage.
    task_publish_retry=False,
    broker_transport_options={
        "socket_connect_timeout": 2,
        "socket_timeout": 2,
        "retry_on_timeout": False,
    },
    beat_schedule={
        "purge-old-data": {
            "task": "maintenance.purge",
            "schedule": crontab(hour=3, minute=0),
        },
    },
)
