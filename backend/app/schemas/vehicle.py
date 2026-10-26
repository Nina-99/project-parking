# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Schemas for a vehicle and a search hit."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class VehicleCreate(BaseModel):
    license_plate: str | None = None
    vehicle_type: str | None = None
    color: str | None = None


class VehicleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    license_plate: str | None
    vehicle_type: str | None
    color: str | None
    created_at: datetime


class VehicleSearchHit(BaseModel):
    """One candidate in an incident search.

    ``match`` tells the operator why the row is there: ``exact`` is the colour
    they asked for, ``neighbour`` is a measured confusion neighbour returned
    because the classifier cannot separate them reliably.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    license_plate: str | None
    vehicle_type: str | None
    color: str | None
    color_confidence: float | None
    image_url: str | None
    created_at: datetime
    match: Literal["exact", "neighbour"]


class VehicleSession(BaseModel):
    id: uuid.UUID
    entry_gate: str | None
    exit_gate: str | None
    entry_time: datetime
    exit_time: datetime | None
    duration_seconds: int | None
    status: str
    close_reason: str | None = None
    closed_by: str | None = None


class VehicleDetail(BaseModel):
    id: uuid.UUID
    license_plate: str | None
    vehicle_type: str | None
    vehicle_type_confidence: float | None
    color: str | None
    color_confidence: float | None
    image_url: str | None
    created_at: datetime
    sessions: list[VehicleSession]
