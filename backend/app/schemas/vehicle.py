"""Pydantic schemas for Vehicle."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VehicleCreate(BaseModel):
    license_plate: str | None = None
    vehicle_type: str | None = None
    color: str | None = None
    brand: str | None = None


class VehicleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    license_plate: str | None
    vehicle_type: str | None
    color: str | None
    brand: str | None
    created_at: datetime
