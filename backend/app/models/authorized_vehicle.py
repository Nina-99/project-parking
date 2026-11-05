# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Authorized vehicle model for access control (RF-10)."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class AuthorizedVehicle(Base):
    __tablename__ = "authorized_vehicles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    license_plate: Mapped[str] = mapped_column(
        String(20), unique=True, index=True, nullable=False
    )
    owner_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    vehicle_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    color: Mapped[str | None] = mapped_column(String(30), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False, server_default="true"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<AuthorizedVehicle {self.license_plate} active={self.is_active}>"
