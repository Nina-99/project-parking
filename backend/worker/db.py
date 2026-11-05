# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Backwards-compatible import for the worker.

The engine lives in the database adapter now, because the API also needs it for
the manual close path. This module keeps ``backend.worker.db`` working for
existing callers and for the runbook.
"""
from __future__ import annotations

from backend.adapters.outbound.db.session import get_engine, get_session_factory, sync_session

__all__ = ["get_engine", "get_session_factory", "sync_session"]
