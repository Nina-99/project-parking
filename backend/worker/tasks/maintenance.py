# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Operational tasks: retention.

Vehicles and sessions are the record of what happened and are kept. Telemetry
and crops are evidence for model health and incident thumbnails; after
``RETENTION_DAYS`` they expire. A missing image is honest; a full disk is not.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from celery import shared_task
from sqlalchemy import delete
from sqlalchemy.engine import CursorResult

from backend.adapters.outbound.db.session import sync_session
from backend.adapters.outbound.storage.clips import ClipStore
from backend.adapters.outbound.storage.crops import CropFileStore
from backend.app.core.config import settings
from backend.app.models.detection_event import DetectionEvent


@shared_task(name="maintenance.purge")
def purge_old_data(retention_days: int | None = None) -> dict[str, int]:
    days = settings.RETENTION_DAYS if retention_days is None else retention_days
    cutoff = datetime.now(UTC) - timedelta(days=days)

    with sync_session() as session:
        result = cast(
            CursorResult[Any],
            session.execute(
                delete(DetectionEvent).where(DetectionEvent.frame_timestamp < cutoff),
            ),
        )
        deleted_events = result.rowcount or 0

    deleted_crops = CropFileStore(settings.ENTRY_IMAGE_DIR).purge(cutoff)
    deleted_clips = ClipStore(
        Path(settings.ENTRY_IMAGE_DIR) / "clips"
    ).purge(cutoff)
    return {
        "deleted_events": deleted_events,
        "deleted_crops": deleted_crops,
        "deleted_clips": deleted_clips,
    }
