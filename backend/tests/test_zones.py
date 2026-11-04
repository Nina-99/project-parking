# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The access polygon: what counts as a vehicle at the gate.

Without this rule the worker turns street traffic into sessions -- one measured
hour produced 129 sessions against 7 crossings -- so these tests fix the boundary
between "seen" and "entered".
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.adapters.inbound.worker import tracking
from backend.app.core.config import settings
from backend.domain.events import GateDetection, TrackAggregator
from backend.domain.vehicle import VehicleAttributes
from backend.domain.zones import (
    ZoneGate,
    bottom_centre,
    point_in_polygon,
    side_of_line,
)

WIDTH, HEIGHT = 1280, 720
# Apron below 60% of the frame; the line at 50%. The gap is deliberate: on the
# real cameras the polygon is deeper than the tripwire, and that gap is what
# separates traffic grazing the line from a vehicle that actually entered.
INSIDE = [[0.0, 0.6], [1.0, 0.6], [1.0, 1.0], [0.0, 1.0]]
TRIPWIRE = [[0.0, 0.5], [1.0, 0.5]]

ATTRIBUTES = VehicleAttributes(vehicle_type="car", vehicle_type_confidence=0.9,
                               color="gris", color_confidence=0.8)
START = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


def box_at(bottom_y: int, x: int = 400) -> tuple[int, int, int, int]:
    return (x, bottom_y - 200, x + 260, bottom_y)


def gate() -> ZoneGate:
    return ZoneGate.from_normalised(INSIDE, TRIPWIRE, WIDTH, HEIGHT)


def test_bottom_centre_is_where_the_tyres_meet_the_ground() -> None:
    assert bottom_centre((100, 50, 300, 250)) == (200.0, 250.0)


def test_a_point_inside_and_outside_a_square() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert point_in_polygon((5.0, 5.0), square)
    assert not point_in_polygon((15.0, 5.0), square)


def test_side_of_line_distinguishes_the_two_halves() -> None:
    line = [(0.0, 5.0), (10.0, 5.0)]
    assert side_of_line((5.0, 8.0), line) == 1
    assert side_of_line((5.0, 2.0), line) == -1


def feed(aggregator: TrackAggregator, bottoms: list[int]) -> list:
    produced = []
    for step, bottom in enumerate(bottoms):
        produced += aggregator.update("gate-1", [GateDetection(box_at(bottom), ATTRIBUTES)],
                                      at(step * 0.5))
    return produced


def test_a_vehicle_parked_in_the_street_is_not_inside() -> None:
    assert not gate().in_inside(box_at(bottom_y=270))


def test_a_vehicle_on_the_apron_is_inside() -> None:
    assert gate().in_inside(box_at(bottom_y=500))


def test_the_apron_is_on_the_side_the_geometry_says() -> None:
    # The inside zone sits below the line, so its side is +1.
    assert gate().inside_side == 1


def test_a_track_that_never_crosses_emits_nothing() -> None:
    assert feed(TrackAggregator(gate=gate()), [270] * 6) == []


def test_a_track_staying_inside_never_crossed_and_emits_nothing() -> None:
    """A parked car inside the apron is not an event: it did not cross."""
    assert feed(TrackAggregator(gate=gate()), [500] * 6) == []


def test_crossing_towards_the_apron_is_an_entry() -> None:
    # Gradual enough for the tracker to keep one track: a jump breaks IoU match.
    produced = feed(TrackAggregator(gate=gate()),
                    [270, 300, 330, 360, 390, 420, 450, 480])
    assert len(produced) == 1
    assert produced[0].direction == "entry"


def test_crossing_away_from_the_apron_is_an_exit() -> None:
    produced = feed(TrackAggregator(gate=gate()),
                    [480, 450, 420, 390, 360, 330, 300, 270])
    assert len(produced) == 1
    assert produced[0].direction == "exit"


def test_grazing_the_line_without_reaching_the_apron_is_not_an_event() -> None:
    """The street runs along the tripwire; passing over it is not entering."""
    produced = feed(TrackAggregator(gate=gate()),
                    [300, 330, 370, 400, 410, 415, 410, 400])
    assert produced == []


def test_a_single_grazing_sample_is_not_a_crossing() -> None:
    produced = feed(TrackAggregator(gate=gate(), crossing_hits=2),
                    [300, 330, 370, 340, 310, 280])
    assert produced == []


def test_a_track_is_reported_after_it_crosses_even_if_confirmed_earlier() -> None:
    produced = feed(TrackAggregator(gate=gate()),
                    [270, 300, 330, 360, 390, 420, 450, 480, 500])
    assert len(produced) == 1
    assert produced[0].direction == "entry"


def test_without_a_gate_the_old_behaviour_is_kept() -> None:
    produced = feed(TrackAggregator(), [270] * 4)
    assert len(produced) == 1
    assert produced[0].direction is None


@pytest.fixture
def zones_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"cameras": {"channel16": {
        "inside_zone": INSIDE, "tripwire": TRIPWIRE}}}), encoding="utf-8")
    monkeypatch.setattr(settings, "ZONES_CONFIG_PATH", str(path))
    monkeypatch.setattr(settings, "CAMERA_ZONES", "gate-1=channel16,gate-2=channel6")
    tracking.reset_cache()
    yield path
    tracking.reset_cache()


@pytest.mark.usefixtures("zones_file")
def test_the_adapter_maps_the_capture_camera_to_its_zone_key() -> None:
    loaded = tracking.gate_for("gate-1", WIDTH, HEIGHT)
    assert loaded is not None
    assert loaded.in_inside(box_at(500))


@pytest.mark.usefixtures("zones_file")
def test_a_camera_without_zones_gets_no_gate() -> None:
    """gate-2 is named in the mapping but absent from the file."""
    assert tracking.gate_for("gate-2", WIDTH, HEIGHT) is None


@pytest.mark.usefixtures("zones_file")
def test_the_aggregator_receives_the_gate() -> None:
    aggregator = tracking.aggregator_for("gate-1", WIDTH, HEIGHT)
    assert aggregator.gate is not None
