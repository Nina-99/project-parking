# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Staleness is a function of the clock, not a stored state."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backend.domain.session import age_minutes, is_stale

ENTRY = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


def test_a_session_is_not_stale_before_the_policy_window() -> None:
    assert not is_stale(ENTRY, ENTRY + timedelta(hours=11, minutes=59), 12)


def test_a_session_is_stale_at_the_window() -> None:
    assert is_stale(ENTRY, ENTRY + timedelta(hours=12), 12)


def test_a_naive_timestamp_is_assumed_to_be_utc() -> None:
    naive_entry = ENTRY.replace(tzinfo=None)

    assert not is_stale(naive_entry, ENTRY + timedelta(hours=1), 12)


def test_age_is_reported_in_whole_minutes_and_never_negative() -> None:
    assert age_minutes(ENTRY, ENTRY + timedelta(minutes=90, seconds=30)) == 90
    assert age_minutes(ENTRY, ENTRY - timedelta(minutes=5)) == 0


def test_a_non_positive_window_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        is_stale(ENTRY, ENTRY, 0)
