# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""What the Celery task is made of.

Building this in one place is what makes the task testable: the task receives a
fully wired object graph instead of reaching for `DetectionEngine.get_instance`
and the global session factory inside its body.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from backend.adapters.inbound.worker.clips import ClipRecorder
from backend.adapters.inbound.worker.ports import CropStore, VehicleAnalyzer
from backend.adapters.inbound.worker.tracking import aggregator_for
from backend.adapters.outbound.bus.redis_publisher import RedisEventPublisher
from backend.adapters.outbound.db.session import get_session_factory
from backend.adapters.outbound.db.unit_of_work import SqlAlchemyUnitOfWork
from backend.adapters.outbound.ml.plate_reader import build_plate_reader
from backend.adapters.outbound.storage.clips import ClipStore
from backend.adapters.outbound.storage.crops import CropFileStore
from backend.adapters.outbound.time import SystemClock
from backend.app.core.config import settings
from backend.domain.events import TrackAggregator
from backend.domain.ports import Clock, EventPublisher, PlateReader, UnitOfWork


@dataclass(frozen=True)
class WorkerDependencies:
    analyzer: VehicleAnalyzer
    clock: Clock
    uow_factory: Callable[[], AbstractContextManager[UnitOfWork]]
    publisher: EventPublisher
    crop_store: CropStore
    aggregator_for: Callable[[str, int, int], TrackAggregator]
    plate_reader: PlateReader | None = None
    model_name: str = "unknown"
    clips: ClipRecorder | None = None


# One recorder for the process: a clip spans several tasks, so building a new
# one per frame would lose the "before" buffer and leave every clip empty.
_clip_recorder: ClipRecorder | None = None


def clip_recorder() -> ClipRecorder:
    global _clip_recorder
    if _clip_recorder is None:
        _clip_recorder = ClipRecorder(
            ClipStore(Path(settings.ENTRY_IMAGE_DIR) / "clips")
        )
    return _clip_recorder


def build_worker_dependencies() -> WorkerDependencies:
    """The production graph. Models load on first use, not at import."""
    from backend.worker.engine import DetectionEngine

    return WorkerDependencies(
        # cv2 frames are uint8; the engine annotates a plain ndarray, which mypy
        # cannot match invariantly. The cast records what the types cannot.
        analyzer=cast(VehicleAnalyzer, DetectionEngine.get_instance()),
        clock=SystemClock(),
        uow_factory=lambda: SqlAlchemyUnitOfWork(get_session_factory()),
        publisher=RedisEventPublisher(
            settings.CELERY_BROKER_URL, settings.EVENTS_CHANNEL,
        ),
        crop_store=CropFileStore(settings.ENTRY_IMAGE_DIR),
        aggregator_for=aggregator_for,
        plate_reader=build_plate_reader(),
        model_name=Path(settings.YOLO_MODEL_PATH).stem,
        clips=clip_recorder(),
    )
