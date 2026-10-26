# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The worker handler, exercised without a GPU, a broker or a database."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from backend.adapters.inbound.worker.dependencies import WorkerDependencies
from backend.adapters.inbound.worker.handler import handle_frame
from backend.domain.events import Box, TrackAggregator
from backend.domain.vehicle import Plate
from backend.tests.fakes import (
    FakePublisher,
    FakeUnitOfWork,
    FixedClock,
    open_session,
)
from backend.worker.tasks.detection import parse_captured_at

FRAME = np.zeros((120, 120, 3), dtype=np.uint8)
BASE = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


@dataclass(frozen=True)
class FakeDetection:
    box: Box = (0, 0, 100, 100)
    detection_confidence: float = 0.91
    vehicle_type: str | None = "car"
    vehicle_type_confidence: float | None = 0.9
    color: str | None = "plata"
    color_confidence: float | None = 0.8


class FakeAnalyzer:
    def __init__(self, detections: list[FakeDetection]) -> None:
        self._detections = detections

    def process(self, _frame: np.ndarray) -> list[FakeDetection]:
        return self._detections


class FakeCrops:
    def __init__(self) -> None:
        self.saved: list[tuple[Box, str]] = []

    def save(self, _frame: np.ndarray, box: Box, camera_id: str) -> str | None:
        self.saved.append((box, camera_id))
        return "/tmp/crop.jpg"


class FakePlateReader:
    def __init__(self, plate: Plate | None) -> None:
        self.plate = plate
        self.reads = 0

    def read(self, _crop: np.ndarray) -> Plate | None:
        self.reads += 1
        return self.plate


def build(
    uow: FakeUnitOfWork,
    detections: list[FakeDetection] | None = None,
    clock: FixedClock | None = None,
    plate_reader: FakePlateReader | None = None,
) -> tuple[WorkerDependencies, FakeCrops]:
    crops = FakeCrops()
    aggregators: dict[str, TrackAggregator] = {}
    dependencies = WorkerDependencies(
        analyzer=FakeAnalyzer(detections if detections is not None else [FakeDetection()]),
        clock=clock or FixedClock(),
        uow_factory=lambda: uow,
        publisher=FakePublisher(),
        crop_store=crops,
        aggregator_for=lambda camera_id: aggregators.setdefault(
            camera_id, TrackAggregator(),
        ),
        plate_reader=plate_reader,
        model_name="yolo26x",
    )
    return dependencies, crops


def feed(
    dependencies: WorkerDependencies,
    frames: int,
    *,
    start: datetime = BASE,
    camera_id: str = "gate-1",
    gate: str = "entry",
) -> list[dict[str, object]]:
    return [
        handle_frame(
            dependencies,
            camera_id=camera_id,
            frame=FRAME,
            gate=gate,
            captured_at=start + timedelta(seconds=index * 0.5),
        )
        for index in range(frames)
    ]


def test_three_frames_of_one_car_create_one_vehicle_and_one_session() -> None:
    uow = FakeUnitOfWork()
    dependencies, crops = build(uow)

    results = feed(dependencies, frames=3)

    assert results[0]["events"] == []
    assert results[1]["events"] == []
    assert len(results[2]["events"]) == 1
    assert len(uow.vehicles.added) == 1
    assert uow.sessions.opened == [("vehicle-1", "gate-1", BASE + timedelta(seconds=1))]
    assert crops.saved == [((0, 0, 100, 100), "gate-1")]


def test_the_session_time_is_when_the_frame_was_sampled_not_when_the_task_ran() -> None:
    uow = FakeUnitOfWork()
    dependencies, _ = build(uow, clock=FixedClock())  # clock says 10:00

    sampled = datetime(2026, 5, 4, 8, 30, tzinfo=UTC)
    feed(dependencies, frames=3, start=sampled)

    assert uow.sessions.opened[0][2] == sampled + timedelta(seconds=1)


def test_a_naive_capture_time_is_treated_as_utc() -> None:
    uow = FakeUnitOfWork()
    dependencies, _ = build(uow)

    feed(dependencies, frames=3, start=datetime(2026, 5, 4, 8, 30))

    entry = uow.sessions.opened[0][2]
    assert entry.tzinfo is not None
    assert entry == datetime(2026, 5, 4, 8, 30, 1, tzinfo=UTC)


def test_a_frame_without_vehicles_does_nothing() -> None:
    uow = FakeUnitOfWork()
    dependencies, _ = build(uow, detections=[])

    result = handle_frame(
        dependencies,
        camera_id="gate-1",
        frame=FRAME,
        gate="entry",
        captured_at=BASE,
    )

    assert result["detections"] == 0
    assert result["events"] == []
    assert uow.vehicles.added == []


def test_two_vehicles_in_one_frame_produce_two_events() -> None:
    uow = FakeUnitOfWork()
    dependencies, _ = build(uow, detections=[
        FakeDetection(box=(0, 0, 100, 100)),
        FakeDetection(box=(500, 0, 600, 100), color="negro"),
    ])

    feed(dependencies, frames=3)

    assert len(uow.vehicles.added) == 2
    assert len(uow.sessions.opened) == 2


def test_parse_captured_at_accepts_iso_and_rejects_garbage() -> None:
    assert parse_captured_at("2026-05-04T10:00:00+00:00") == BASE
    assert parse_captured_at(None) is None
    assert parse_captured_at("not-a-date") is None


def test_a_read_plate_is_persisted_on_entry() -> None:
    uow = FakeUnitOfWork()
    plate = Plate(text="1234ABC", confidence=0.93)
    dependencies, _ = build(uow, plate_reader=FakePlateReader(plate))

    feed(dependencies, frames=3)

    assert uow.vehicles.added[0][1] == plate


def test_a_read_plate_closes_the_session_on_exit() -> None:
    uow = FakeUnitOfWork(active=[open_session(plate="1234ABC")])
    reader = FakePlateReader(Plate(text="1234ABC", confidence=0.93))
    dependencies, _ = build(uow, plate_reader=reader)

    feed(dependencies, frames=3, gate="exit")

    assert reader.reads == 1
    assert uow.sessions.closed[0]["reason"] == "plate_match"
    assert uow.sessions.closed[0]["operator"] is None


def test_without_a_reader_no_plate_is_recorded() -> None:
    uow = FakeUnitOfWork()
    dependencies, _ = build(uow)

    feed(dependencies, frames=3)

    assert uow.vehicles.added[0][1] is None


def test_the_handler_records_the_inference_that_produced_the_event() -> None:
    uow = FakeUnitOfWork()
    dependencies, _ = build(uow)

    feed(dependencies, frames=3)

    record = uow.detections.records[0]
    assert record["model_name"] == "yolo26x"
    assert record["inference_latency_ms"] is not None
    assert record["inference_latency_ms"] >= 0
    assert record["detection_confidence"] == 0.91
