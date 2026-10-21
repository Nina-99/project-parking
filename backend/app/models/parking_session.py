"""
ParkingSession model — entry/exit records for each parking visit.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base

import enum


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ParkingSession(Base):
    __tablename__ = "parking_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    vehicle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("vehicles.id"), nullable=False, index=True,
    )

    # Gate info
    entry_gate: Mapped[str | None] = mapped_column(String(20), nullable=True)
    exit_gate: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Timestamps
    entry_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    exit_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Duration in seconds (computed)
    duration_seconds: Mapped[int | None] = mapped_column(nullable=True)

    # Status
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status"),
        default=SessionStatus.ACTIVE,
        index=True,
    )

    # Images at exit
    exit_image_path: Mapped[str | None] = mapped_column(nullable=True)

    # Relationship
    vehicle: Mapped["Vehicle"] = relationship(back_populates="sessions")

    def __repr__(self) -> str:
        return f"<ParkingSession {self.id} [{self.status.value}]>"
