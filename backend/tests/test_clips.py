# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Event clips: the buffer across frames, the file on disk, and the handler hook."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from backend.adapters.inbound.worker.clips import ClipRecorder
from backend.adapters.inbound.worker.dependencies import WorkerDependencies
from backend.adapters.inbound.worker.handler import handle_frame
from backend.adapters.outbound.storage.clips import (
    ClipFrame,
    ClipStore,
    resolve_clip_path,
)
from backend.domain.events import Box, TrackAggregator
from backend.tests.fakes import FakePublisher, FakeUnitOfWork, FixedClock

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)
FRAME = np.zeros((72, 128, 3), dtype=np.uint8)


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int, datetime, datetime]] = []

    def save(self, vehicle_id: str, frames: list[ClipFrame]) -> str | None:
        self.saved.append(
            (vehicle_id, len(frames), frames[0].at, frames[-1].at)
        )
        return f"/tmp/{vehicle_id}.webm"


def test_the_clip_keeps_the_before_and_after_windows() -> None:
    store = FakeStore()
    recorder = ClipRecorder(store, before=2.0, after=2.0, buffer_seconds=2.5)

    for index in (-4, -3, -2, -1, 0):
        recorder.feed("gate-1", FRAME, NOW + timedelta(seconds=index * 0.5))

    recorder.start("gate-1", "vehicle-1", NOW)
    for index in (1, 2, 3, 4, 5):
        recorder.feed("gate-1", FRAME, NOW + timedelta(seconds=index * 0.5))

    vehicle_id, count, first, last = store.saved[0]
    assert vehicle_id == "vehicle-1"
    assert first == NOW - timedelta(seconds=2.0), "the before window starts at -2s"
    assert last == NOW + timedelta(seconds=2.0), "nothing after the deadline is kept"
    assert count == 9


def test_a_clip_waits_for_frames_after_the_event() -> None:
    store = FakeStore()
    recorder = ClipRecorder(store, before=2.0, after=2.0)

    recorder.feed("gate-1", FRAME, NOW)
    recorder.start("gate-1", "vehicle-1", NOW)

    assert store.saved == [], "a clip is not written while it is still open"

    recorder.feed("gate-1", FRAME, NOW + timedelta(seconds=3.0))

    assert len(store.saved) == 1


def test_a_real_clip_is_written_and_purged(tmp_path: Path) -> None:
    store = ClipStore(tmp_path / "clips")
    frames = [ClipFrame(NOW, FRAME) for _ in range(3)]

    path = store.save("vehicle-1", frames)

    assert path is not None
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 0
    assert resolve_clip_path("vehicle-1", media_root=tmp_path) is not None

    os.utime(path, (0, 0))
    assert store.purge(NOW) == 1
    assert not Path(path).exists()


def test_one_frame_is_not_a_clip(tmp_path: Path) -> None:
    store = ClipStore(tmp_path / "clips")

    assert store.save("vehicle-1", [ClipFrame(NOW, FRAME)]) is None


class FakeClips:
    def __init__(self) -> None:
        self.fed: list[tuple[str, datetime]] = []
        self.started: list[tuple[str, str, datetime]] = []

    def feed(self, camera_id: str, _frame: np.ndarray, at: datetime) -> None:
        self.fed.append((camera_id, at))

    def start(self, camera_id: str, vehicle_id: str, at: datetime) -> None:
        self.started.append((camera_id, vehicle_id, at))


class FakeCrops:
    def save(self, _frame: np.ndarray, _box: object, _camera_id: str) -> str | None:
        return "/tmp/crop.jpg"


def test_the_handler_starts_a_clip_when_it_confirms_an_event() -> None:
    clips = FakeClips()
    aggregators: dict[str, TrackAggregator] = {}

    class Analyzer:
        def process(self, _frame: np.ndarray) -> list[object]:
            return [type("D", (), {
                "box": Box((0, 0, 100, 100)),  # type: ignore[misc]
                "detection_confidence": 0.9,
                "vehicle_type": "car",
                "vehicle_type_confidence": 0.9,
                "color": "plata",
                "color_confidence": 0.8,
            })()]

    dependencies = WorkerDependencies(
        analyzer=Analyzer(),  # type: ignore[arg-type]
        clock=FixedClock(),
        uow_factory=lambda: FakeUnitOfWork(),
        publisher=FakePublisher(),
        crop_store=FakeCrops(),
        aggregator_for=lambda camera_id, _w, _h: aggregators.setdefault(
            camera_id, TrackAggregator(),
        ),
        clips=clips,
    )

    for index in range(3):
        handle_frame(
            dependencies,
            camera_id="gate-1",
            frame=FRAME,
            gate="entry",
            captured_at=NOW + timedelta(seconds=index * 0.5),
        )

    assert len(clips.fed) == 3
    assert clips.started == [("gate-1", "vehicle-1", NOW + timedelta(seconds=1.0))]
