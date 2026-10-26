# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Occupancy as a percentage of a configured capacity.

Capacity is configuration, not a table: the scope is total occupancy at the
gates, not a map of individual spaces.
"""
from __future__ import annotations


def occupancy_percent(active: int, capacity: int) -> float | None:
    """``None`` when no capacity is configured; over 100% is reported as is."""
    if capacity <= 0:
        return None
    return round(active * 100.0 / capacity, 1)
