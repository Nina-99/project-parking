# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Per-camera clip recorder, stateful across frames.

The worker sees one frame per task, so the "before" of a crossing has to be kept
alive between calls. This module keeps a short ring buffer per camera and, when
the tracker confirms an event, opens a clip that is seeded with that buffer and
completed by the frames that follow. It is the same process-local assumption as
the track aggregator: one worker process per camera queue.

Encoding happens when the clip closes, not on the frame path that produced it,
and a failure returns None instead of raising: losing evidence is bad, stopping
the detector is worse.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from backend.adapters.inbound.worker.ports import Frame
from backend.adapters.outbound.storage.clips import ClipFrame, ClipStore, prepare

logger = logging.getLogger(__name__)

BEFORE_SECONDS = 2.0
AFTER_SECONDS = 2.0
BUFFER_SECONDS = 2.5


@dataclass
class _OpenClip:
    vehicle_id: str
    deadline: datetime
    shape: tuple[int, ...]
    frames: list[ClipFrame] = field(default_factory=list)


class ClipRecorder:
    def __init__(
        self,
        store: ClipStore,
        *,
        before: float = BEFORE_SECONDS,
        after: float = AFTER_SECONDS,
        buffer_seconds: float = BUFFER_SECONDS,
    ) -> None:
        if before <= 0 or after <= 0:
            raise ValueError("before and after must be positive")
        if buffer_seconds < before:
            raise ValueError("buffer_seconds must cover the before window")
        self.store = store
        self.before = before
        self.after = after
        self.buffer_seconds = buffer_seconds
        self._buffers: dict[str, deque[ClipFrame]] = {}
        self._open: dict[str, list[_OpenClip]] = {}

    def feed(self, camera_id: str, frame: Frame, at: datetime) -> None:
        """Keep the ring buffer current and advance every open clip."""
        prepared = prepare(frame)
        buffer = self._buffers.setdefault(camera_id, deque())
        buffer.append(ClipFrame(at, prepared))

        cutoff = at - timedelta(seconds=self.buffer_seconds)
        while buffer and buffer[0].at < cutoff:
            buffer.popleft()

        for clip in list(self._open.get(camera_id, [])):
            if at > clip.deadline:
                self._finish(camera_id, clip)
            elif prepared.shape == clip.shape:
                clip.frames.append(ClipFrame(at, prepared))

    def start(self, camera_id: str, vehicle_id: str, at: datetime) -> None:
        """Open a clip for a confirmed event, seeded with what came before."""
        buffer = self._buffers.get(camera_id)
        if not buffer:
            return

        window_start = at - timedelta(seconds=self.before)
        seed = [
            item for item in buffer
            if window_start <= item.at <= at
        ]
        if not seed:
            return

        shape = seed[-1].image.shape
        seed = [item for item in seed if item.image.shape == shape]
        self._open.setdefault(camera_id, []).append(
            _OpenClip(
                vehicle_id=vehicle_id,
                deadline=at + timedelta(seconds=self.after),
                shape=shape,
                frames=seed,
            )
        )

    def flush(self) -> None:
        """Close every open clip. Tests and shutdown only."""
        for camera_id in list(self._open):
            for clip in list(self._open.get(camera_id, [])):
                self._finish(camera_id, clip)

    def _finish(self, camera_id: str, clip: _OpenClip) -> None:
        open_clips = self._open.get(camera_id)
        if open_clips and clip in open_clips:
            open_clips.remove(clip)
        path = self.store.save(clip.vehicle_id, clip.frames)
        if path is None:
            logger.warning(
                "clip_discarded vehicle=%s frames=%d",
                clip.vehicle_id, len(clip.frames),
            )
        else:
            logger.info(
                "clip_saved vehicle=%s frames=%d path=%s",
                clip.vehicle_id, len(clip.frames), path,
            )
