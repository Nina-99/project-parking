# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""A manual close commits, then announces, and returns the audited outcome."""
from __future__ import annotations

from backend.application.close_session import close_session
from backend.tests.fakes import NOW, FakePublisher, FakeUnitOfWork, FixedClock


def test_a_manual_close_commits_and_publishes_in_that_order() -> None:
    uow = FakeUnitOfWork()
    publisher = FakePublisher(uow.log)

    outcome = close_session(
        session_id="session-1",
        operator="ana",
        reason="left without a read",
        uow=uow,
        publisher=publisher,
        clock=FixedClock(),
    )

    assert uow.log == ["commit", "publish"]
    assert outcome.session_id == "session-1"
    assert outcome.duration_seconds == 3600
    assert outcome.operator == "ana"


def test_the_closed_row_keeps_who_and_why() -> None:
    uow = FakeUnitOfWork()

    close_session(
        session_id="session-1",
        operator="ana",
        reason="left without a read",
        uow=uow,
        publisher=FakePublisher(),
        clock=FixedClock(),
    )

    assert uow.sessions.closed == [{
        "session_id": "session-1",
        "at": NOW,
        "gate": None,
        "reason": "left without a read",
        "operator": "ana",
    }]


def test_the_event_says_the_session_was_closed_by_a_human() -> None:
    publisher = FakePublisher()

    close_session(
        session_id="session-1",
        operator="ana",
        reason="manual",
        uow=FakeUnitOfWork(),
        publisher=publisher,
        clock=FixedClock(),
    )

    payload = publisher.payloads[0]
    assert payload["type"] == "session_closed"
    assert payload["operator"] == "ana"
    assert payload["session_id"] == "session-1"
