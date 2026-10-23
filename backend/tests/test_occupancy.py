"""Occupancy is a count and a denominator; nothing else."""
from __future__ import annotations

from backend.domain.occupancy import occupancy_percent


def test_without_a_configured_capacity_there_is_no_percentage() -> None:
    assert occupancy_percent(5, 0) is None


def test_the_percentage_is_rounded_to_one_decimal() -> None:
    assert occupancy_percent(10, 40) == 25.0
    assert occupancy_percent(1, 3) == 33.3


def test_over_capacity_is_reported_as_is() -> None:
    assert occupancy_percent(50, 40) == 125.0
