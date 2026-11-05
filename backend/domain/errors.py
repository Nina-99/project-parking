# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Domain failures the API has to translate into HTTP.

They are domain errors, not HTTP errors: a session that is not active cannot be
closed means the same thing whether the caller is an endpoint, a task or a test.
"""
from __future__ import annotations


class SessionNotFoundError(Exception):
    """No session with that id exists."""


class SessionNotActiveError(Exception):
    """The session was already closed; closing it again would rewrite history."""


class VehicleNotFoundError(Exception):
    """No vehicle with that id exists."""


class InvalidValidationDecisionError(Exception):
    """The operator's decision is not one the domain accepts.

    A corrected plate that is not a plate belongs here too: accepting it would
    store an identifier the matching policy can never use.
    """
