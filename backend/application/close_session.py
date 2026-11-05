# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""A human closes a session the model could not close.

This is the operational answer while plate matching is not reliable: the count
is corrected by someone who can see the lot, with a reason and a name attached.
The alternative -- guessing a match from colour and time -- would empty the lot
silently, which is worse than a visible number that a person can fix.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.domain.ports import Clock, EventPublisher, UnitOfWork


@dataclass(frozen=True)
class CloseOutcome:
    session_id: str
    duration_seconds: int
    reason: str
    operator: str


def close_session(
    *,
    session_id: str,
    operator: str,
    reason: str,
    uow: UnitOfWork,
    publisher: EventPublisher,
    clock: Clock,
) -> CloseOutcome:
    """Close an active session by hand, after the commit is durable."""
    now = clock.now()
    duration = uow.sessions.close(
        session_id,
        now,
        reason=reason,
        operator=operator,
    )
    uow.commit()

    publisher.publish({
        "type": "session_closed",
        "session_id": session_id,
        "operator": operator,
        "reason": reason,
        "timestamp": now.isoformat(),
    })

    return CloseOutcome(
        session_id=session_id,
        duration_seconds=duration,
        reason=reason,
        operator=operator,
    )
