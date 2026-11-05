# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""One row per camera the system expects.

Heartbeats stay in Redis because they expire on their own (see
``adapters/outbound/cache/camera_status.py``). This table is the other half: the
list of cameras that *should* be reporting. Without it the UI cannot tell "there
are no cameras" from "both cameras are down", because a camera that never sent a
heartbeat has nothing in Redis to list.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    # The identifier the capturer publishes heartbeats under, and the one the
    # worker tags events with. Not a surrogate key: it must match exactly.
    camera_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True,
    )
    gate: Mapped[str] = mapped_column(String(20), nullable=False, default="entry")
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    rtsp_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
