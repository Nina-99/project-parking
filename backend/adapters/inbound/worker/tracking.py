# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Process-local track state, one aggregator per camera.

Tracking is stateful: the aggregator has to see the frames of a camera in order.
The worker therefore runs with ``--concurrency=1`` and one process per queue. If
the task is ever fanned out across processes, this registry has to move to a
shared store or the same car will be counted once per process.

Each aggregator also receives the camera's access geometry. A camera whose field
of view includes the street confirms dozens of vehicles an hour that never cross
the gate; without the polygon they all become sessions.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.config import settings
from backend.domain.events import TrackAggregator
from backend.domain.zones import ZoneGate

_aggregators: dict[str, TrackAggregator] = {}
_gates: dict[tuple[str, int, int], ZoneGate | None] = {}


def _zone_key(camera_id: str) -> str:
    """Map a capture camera id to its key in ``zones_config.json``.

    ``capture.py`` names cameras ``gate-1``/``gate-2``; the zones file names them
    after the recorder channel. The mapping lives in configuration because only
    the site knows which door is which.
    """
    for item in settings.CAMERA_ZONES.split(","):
        if "=" not in item:
            continue
        name, key = item.split("=", 1)
        if name.strip() == camera_id:
            return key.strip()
    return camera_id


def gate_for(camera_id: str, width: int, height: int) -> ZoneGate | None:
    """The access polygon for this camera, or None when the site has none.

    Missing geometry is not an error: the worker keeps working with the old
    behaviour, and the session count is then a count of vehicles *seen*, not of
    vehicles that entered. The warning is emitted once per camera.
    """
    cached = _gates.get((camera_id, width, height), "missing")
    if cached != "missing":
        return cached  # type: ignore[return-value]

    path = Path(settings.ZONES_CONFIG_PATH)
    gate: ZoneGate | None = None
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        camera = data.get("cameras", {}).get(_zone_key(camera_id), {})
        inside = camera.get("inside_zone") or []
        tripwire = camera.get("tripwire") or []
        if len(inside) >= 3 and len(tripwire) == 2:
            gate = ZoneGate.from_normalised(inside, tripwire, width, height)
    if gate is None:
        print(f"[tracking] no usable zones for {camera_id} in {path}; "
              f"every confirmed vehicle will be reported", flush=True)
    _gates[(camera_id, width, height)] = gate
    return gate


def aggregator_for(camera_id: str, width: int = 1280,
                   height: int = 720) -> TrackAggregator:
    aggregator = _aggregators.get(camera_id)
    if aggregator is None:
        aggregator = TrackAggregator(gate=gate_for(camera_id, width, height))
        _aggregators[camera_id] = aggregator
    return aggregator


def reset_cache() -> None:
    """Drop cached state. Tests and configuration reloads only."""
    _aggregators.clear()
    _gates.clear()
