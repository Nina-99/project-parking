# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Process-local track state, one aggregator per camera.

Tracking is stateful: the aggregator has to see the frames of a camera in order.
The worker therefore runs with ``--concurrency=1`` and one process per queue. If
the task is ever fanned out across processes, this registry has to move to a
shared store or the same car will be counted once per process.
"""
from __future__ import annotations

from backend.domain.events import TrackAggregator

_aggregators: dict[str, TrackAggregator] = {}


def aggregator_for(camera_id: str) -> TrackAggregator:
    if camera_id not in _aggregators:
        _aggregators[camera_id] = TrackAggregator()
    return _aggregators[camera_id]
