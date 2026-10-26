# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Frames become tracks, tracks become events -- once each.

These tests exist because the alternative (persist one vehicle per frame) looks
correct in a five-second demo and corrupts the occupancy count everywhere else.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.domain.events import (
    GateDetection,
    TrackAggregator,
    intersection_over_union,
)
from backend.domain.vehicle import VehicleAttributes

START = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)
CROP = VehicleAttributes(vehicle_type="car", vehicle_type_confidence=0.9,
                         color="plata", color_confidence=0.8)


def at(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


def detection(
    x: int = 0,
    size: int = 100,
    attributes: VehicleAttributes = CROP,
) -> GateDetection:
    return GateDetection(box=(x, 0, x + size, size), attributes=attributes)


def feed(aggregator: TrackAggregator, frames: int, step: float = 0.5) -> list:
    events = []
    for index in range(frames):
        events.extend(aggregator.update("gate-1", [detection()], at(index * step)))
    return events


def test_a_vehicle_seen_in_three_frames_emits_one_event() -> None:
    events = feed(TrackAggregator(), frames=3)

    assert len(events) == 1
    assert events[0].camera_id == "gate-1"
    assert events[0].attributes == CROP


def test_two_frames_are_not_enough_to_call_it_a_vehicle() -> None:
    assert feed(TrackAggregator(), frames=2) == []


def test_further_frames_do_not_emit_the_same_vehicle_again() -> None:
    assert len(feed(TrackAggregator(), frames=20)) == 1


def test_a_car_approaching_keeps_one_track_and_the_largest_view() -> None:
    aggregator = TrackAggregator()
    events = []
    for index in range(5):
        events.extend(
            aggregator.update(
                "gate-1",
                [detection(x=index * 5, size=100 + index * 10)],
                at(index * 0.5),
            )
        )

    assert len(events) == 1, "a moving car is still one car"
    assert events[0].box == (10, 0, 130, 120), "the event carries the largest view"


def test_two_vehicles_in_different_places_emit_two_events() -> None:
    aggregator = TrackAggregator()
    for index in range(3):
        events = aggregator.update(
            "gate-1",
            [detection(x=0), detection(x=500)],
            at(index * 0.5),
        )

    assert len(events) == 2


def test_a_track_that_vanishes_past_max_age_is_a_new_event() -> None:
    aggregator = TrackAggregator(max_age_seconds=2.0)
    assert len(feed(aggregator, frames=3)) == 1

    later = at(10)
    aggregator.update("gate-1", [], later)
    events = []
    for index in range(3):
        events.extend(aggregator.update("gate-1", [detection()], later + timedelta(seconds=index)))

    assert len(events) == 1, "a second appearance is a second event"
    assert events[0].track_id == 2


def test_cameras_do_not_share_tracks() -> None:
    aggregator = TrackAggregator()
    for index in range(3):
        entry = aggregator.update("gate-1", [detection()], at(index * 0.5))
        exit_ = aggregator.update("gate-2", [detection()], at(index * 0.5))

    assert len(entry) == 1
    assert len(exit_) == 1
    assert entry[0].track_id != exit_[0].track_id


def test_the_classification_comes_from_the_clearest_frame() -> None:
    aggregator = TrackAggregator()
    blurry = VehicleAttributes(vehicle_type="car", color="gris", color_confidence=0.4)
    clear = VehicleAttributes(vehicle_type="car", color="plata", color_confidence=0.9)
    sizes = [100, 140, 110]

    for index, (attributes, size) in enumerate(zip([blurry, clear, blurry], sizes, strict=True)):
        events = aggregator.update(
            "gate-1", [detection(size=size, attributes=attributes)], at(index * 0.5),
        )

    assert events[0].attributes.color == "plata"
    assert events[0].attributes.color_confidence == 0.9


def test_iou_is_zero_when_boxes_do_not_touch() -> None:
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
