"""Pydantic schemas for the read side: sessions and active vehicles."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    vehicle_id: uuid.UUID
    entry_gate: str | None
    exit_gate: str | None
    entry_time: datetime
    exit_time: datetime | None
    duration_seconds: int | None
    status: str
    license_plate: str | None = None
    vehicle_type: str | None = None
    color: str | None = None
    close_reason: str | None = None
    closed_by: str | None = None


class ActiveSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    vehicle_id: uuid.UUID
    entry_time: datetime
    entry_gate: str | None
    license_plate: str | None
    vehicle_type: str | None
    color: str | None
    # Derived from the entry time at read time, never written to the row.
    stale: bool = False
    age_minutes: int = 0
