# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Whether a camera is still reporting.

Liveness is derived from the last heartbeat, not stored as a state: a process
that dies cannot write "offline", and the absence of a recent beat is the only
honest signal.
"""
from __future__ import annotations

from datetime import datetime

from backend.domain.session import as_utc


def is_online(
    last_seen: datetime | None,
    now: datetime,
    offline_after_seconds: float,
) -> bool:
    if offline_after_seconds <= 0:
        raise ValueError("offline_after_seconds must be positive")
    if last_seen is None:
        return False
    age = (as_utc(now) - as_utc(last_seen)).total_seconds()
    return age <= offline_after_seconds
