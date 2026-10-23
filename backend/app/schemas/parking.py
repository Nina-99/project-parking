"""Schemas for entry, exit and session responses."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EntryRequest(BaseModel):
    """Body of a POST to the entry gate."""
    gate_id: str = Field(..., description="Gate identifier (e.g. 'gate-1')")
    image: str = Field(..., description="Base64-encoded JPEG image of the vehicle")
    timestamp: datetime | None = None


class EntryResponse(BaseModel):
    session_id: uuid.UUID
    vehicle_id: uuid.UUID | None
    license_plate: str | None
    vehicle_type: str | None
    color: str | None
    status: str = "registered"


class ExitRequest(BaseModel):
    """Body of a POST to the exit gate."""
    gate_id: str
    image: str  # Base64 JPEG
    session_id: uuid.UUID | None = None  # If we know the session
    timestamp: datetime | None = None


class ExitResponse(BaseModel):
    session_id: uuid.UUID
    vehicle_id: uuid.UUID | None
    license_plate: str | None
    entry_time: datetime
    exit_time: datetime | None
    duration_minutes: float | None
    status: str


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


class ActiveSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    vehicle_id: uuid.UUID
    entry_time: datetime
    entry_gate: str | None
    license_plate: str | None
    vehicle_type: str | None
    color: str | None
