# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Raw inference log: one row per YOLO detection."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parking_sessions.id"), nullable=True, index=True,
    )
    # The vehicle this observation produced. An unmatched exit has no session, so
    # this is the only link between the event and the row the operator reviews.
    vehicle_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("vehicles.id"), nullable=True, index=True,
    )

    camera_id: Mapped[str] = mapped_column(String(50), index=True)
    frame_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )

    model_name: Mapped[str] = mapped_column(String(20))  # yolo26n, etc.
    object_class: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)

    # Normalized 0-1.
    bbox: Mapped[dict] = mapped_column(JSONB)

    crop_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    inference_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(20), nullable=True)

    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # What the system decided: "entry", "exit", or None when the camera has no
    # geometry. Stored because the decision is the result being measured; deriving
    # it later from the camera's role would be reading the input, not the output.
    direction: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)

    def __repr__(self) -> str:
        return f"<Detection {self.object_class} ({self.confidence:.2f})>"
