"""One row per vehicle seen at a gate."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.core.database import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    license_plate: Mapped[str | None] = mapped_column(
        String(20), nullable=True, index=True,
    )
    license_plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    vehicle_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True,
    )
    vehicle_type_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    color: Mapped[str | None] = mapped_column(String(30), nullable=True)
    color_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    brand: Mapped[str | None] = mapped_column(String(50), nullable=True)
    brand_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    entry_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    entry_video_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Serialized numpy array, kept for a future ReID model.
    feature_vector: Mapped[bytes | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    )

    sessions: Mapped[list["ParkingSession"]] = relationship(
        back_populates="vehicle", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Vehicle {self.license_plate or self.id}>"
