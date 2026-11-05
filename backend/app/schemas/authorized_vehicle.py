# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AuthorizedVehicleCreate(BaseModel):
    license_plate: str = Field(min_length=3, max_length=20)
    owner_name: str | None = Field(default=None, max_length=120)
    vehicle_type: str | None = Field(default=None, max_length=50)
    color: str | None = Field(default=None, max_length=30)
    notes: str | None = None
    is_active: bool = True


class AuthorizedVehicleUpdate(BaseModel):
    owner_name: str | None = Field(default=None, max_length=120)
    vehicle_type: str | None = Field(default=None, max_length=50)
    color: str | None = Field(default=None, max_length=30)
    notes: str | None = None
    is_active: bool | None = None


class AuthorizedVehicleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    license_plate: str
    owner_name: str | None = None
    vehicle_type: str | None = None
    color: str | None = None
    notes: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
