# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Schemas exposed by the read API."""
from __future__ import annotations

from backend.app.schemas.parking import ActiveSessionResponse, SessionResponse
from backend.app.schemas.vehicle import VehicleCreate, VehicleResponse

__all__ = [
    "ActiveSessionResponse",
    "SessionResponse",
    "VehicleCreate",
    "VehicleResponse",
]
