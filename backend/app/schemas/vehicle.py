"""Pydantic schemas for Vehicle."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

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
    entry_image_path: str | None
    created_at: datetime
    match: Literal["exact", "neighbour"]
