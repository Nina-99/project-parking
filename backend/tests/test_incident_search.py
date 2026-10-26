# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Incident-search behaviour that does not need the database.

The route itself is a thin query on top of these two functions; the decisions
worth pinning down are which colours a query expands to and how candidates are
ordered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from backend.domain.colours import expand_color
from backend.domain.incident import rank_hits


def test_exact_colour_comes_first_and_neighbours_after() -> None:
    assert expand_color("plata") == ("plata", "gris", "blanco")


def test_a_colour_without_measured_confusion_stays_alone() -> None:
    assert expand_color("rojo") == ("rojo",)


def test_no_colour_expands_to_nothing() -> None:
    assert expand_color(None) == ()
    assert expand_color("") == ()


@dataclass
class Hit:
    color: str | None
    created_at: datetime
    name: str = ""


def _hit(name: str, color: str, minute: int) -> Hit:
    return Hit(color=color, created_at=datetime(2026, 5, 4, 10, minute, tzinfo=UTC), name=name)


def test_exact_match_outranks_a_neighbour_even_if_the_neighbour_is_closer() -> None:
    neighbour = _hit("gris", "gris", 0)
    exact = _hit("plata", "plata", 30)
    at = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)

    ranked = rank_hits([neighbour, exact], "plata", at)

    assert [h.name for h in ranked] == ["plata", "gris"]


def test_candidates_are_ordered_by_distance_to_the_remembered_instant() -> None:
    far = _hit("far", "plata", 40)
    near = _hit("near", "plata", 2)
    at = datetime(2026, 5, 4, 10, 3, tzinfo=UTC)

    ranked = rank_hits([far, near], "plata", at)

    assert [h.name for h in ranked] == ["near", "far"]


def test_a_naive_instant_does_not_raise() -> None:
    """Query strings arrive without a timezone; comparisons must still work."""
    row = _hit("row", "plata", 10)
    at = datetime(2026, 5, 4, 10, 0)  # naive on purpose

    assert rank_hits([row], "plata", at) == [row]


def test_without_an_instant_the_newest_vehicle_leads() -> None:
    older = _hit("older", "plata", 1)
    newer = _hit("newer", "plata", 9)

    ranked = rank_hits([older, newer], None, None)

    assert [h.name for h in ranked] == ["newer", "older"]
