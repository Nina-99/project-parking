# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Session policy: a session is open, closed by evidence, or closed by a human.

Staleness is **derived**, not stored. A row does not become stale because time
passed; it becomes stale relative to the moment you look at it. Persisting that
would need a scheduler and would turn a GET into a write.
"""
from __future__ import annotations

from datetime import UTC, datetime


def as_utc(value: datetime) -> datetime:
    """Assume UTC when a timestamp arrives without a zone."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def is_stale(entry_time: datetime, now: datetime, stale_after_hours: float) -> bool:
    """True when the session has been open longer than the policy allows."""
    if stale_after_hours <= 0:
        raise ValueError("stale_after_hours must be positive")
    age_hours = (as_utc(now) - as_utc(entry_time)).total_seconds() / 3600
    return age_hours >= stale_after_hours


def age_minutes(entry_time: datetime, now: datetime) -> int:
    return max(0, int((as_utc(now) - as_utc(entry_time)).total_seconds() // 60))
