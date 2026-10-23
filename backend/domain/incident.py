"""Ranking for the incident search.

An operator remembers "a silver vehicle entered around 10:00". The window alone
narrows the list to a handful of vehicles at this gate (median six per hour), so
the ordering matters more than the filter: exact colour first, then the closest
moment to the remembered instant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


def _as_utc(value: datetime) -> datetime:
    """Assume UTC when a timestamp arrives without a zone (query strings do)."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class _Hit(Protocol):
    color: str | None
    created_at: datetime


def rank_hits(rows: list[_Hit], requested_color: str | None, at: datetime | None) -> list[_Hit]:
    """Exact colour first, then the closest moment to ``at``, then recency."""
    reference = _as_utc(at) if at is not None else None

    def key(row: _Hit) -> tuple[int, float]:
        exact = 0 if not requested_color or row.color == requested_color else 1
        created = _as_utc(row.created_at)
        if reference is None:
            return (exact, -created.timestamp())
        return (exact, abs((created - reference).total_seconds()))

    return sorted(rows, key=key)
