# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Closing a session is a decision that must never be a guess."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.domain.matching import (
    MIN_PLATE_CONFIDENCE,
    ExitVehicle,
    OpenSession,
    find_match,
    normalize_plate,
)


def _session(session_id: str, plate: str | None, minutes: int,
             confidence: float = 0.9) -> OpenSession:
    return OpenSession(
        session_id=session_id, vehicle_id=f"v-{session_id}", plate_confidence=confidence,
        license_plate=plate, entry_time=datetime(2026, 5, 4, 10, 0, tzinfo=UTC)
        + timedelta(minutes=minutes),
    )


def test_plates_compare_ignoring_separators_and_case() -> None:
    assert normalize_plate("1234-abc") == normalize_plate("1234 ABC") == "1234ABC"
    assert normalize_plate(None) == ""
    assert normalize_plate("  ") == ""


def test_an_exact_plate_closes_its_session() -> None:
    sessions = [_session("a", "1234ABC", 0), _session("b", "9999ZZZ", 5)]

    match = find_match(ExitVehicle("1234-abc", 0.9), sessions)

    assert match is not None and match.session_id == "a"


def test_a_readable_plate_with_no_open_session_matches_nothing() -> None:
    assert find_match(ExitVehicle("7777XYZ", 0.9), [_session("a", "1234ABC", 0)]) is None


def test_an_unreadable_plate_matches_nothing() -> None:
    sessions = [_session("a", "1234ABC", 0)]

    assert find_match(ExitVehicle(None, None), sessions) is None
    assert find_match(ExitVehicle("", 0.9), sessions) is None
    assert find_match(ExitVehicle("1234ABC", None), sessions) is None


def test_a_low_confidence_plate_is_not_evidence() -> None:
    sessions = [_session("a", "1234ABC", 0)]

    assert find_match(ExitVehicle("1234ABC", MIN_PLATE_CONFIDENCE - 0.01), sessions) is None
    assert find_match(ExitVehicle("1234ABC", MIN_PLATE_CONFIDENCE), sessions) is not None


def test_a_low_confidence_entry_plate_does_not_attract_matches() -> None:
    sessions = [_session("a", "1234ABC", 0, confidence=0.2)]

    assert find_match(ExitVehicle("1234ABC", 0.9), sessions) is None


def test_the_most_recent_entry_wins_when_a_plate_repeats() -> None:
    sessions = [_session("early", "1234ABC", 0), _session("late", "1234ABC", 30)]

    match = find_match(ExitVehicle("1234ABC", 0.9), sessions)

    assert match is not None and match.session_id == "late"


def test_looking_alike_is_never_enough() -> None:
    """The measured alternative: colour and type are not identity."""
    sessions = [_session("a", "1234ABC", 0)]

    assert find_match(ExitVehicle("9999ZZZ", 0.99), sessions) is None
