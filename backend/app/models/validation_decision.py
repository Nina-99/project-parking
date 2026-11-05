# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""What an operator decided about one gate event, and why.

The gate event is the machine's claim: "this vehicle entered", "this vehicle
left and did not match". The validation decision is the human's answer. Keeping
them apart is what makes the queue auditable: a confirmed event and a corrected
plate are different facts, and the second one has a before and an after.

The row is append-only. A later correction adds another row; it does not rewrite
the first one, because the point of an audit trail is that it cannot be edited.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class ValidationDecision(Base):
    __tablename__ = "validation_decisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    vehicle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("vehicles.id"), nullable=False, index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parking_sessions.id"), nullable=True,
    )

    decision: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    original_plate: Mapped[str | None] = mapped_column(String(20), nullable=True)
    corrected_plate: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Who decided. Not a foreign key: an operator account can be deleted, and an
    # audit row must survive the account that produced it.
    operator: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return f"<ValidationDecision {self.vehicle_id} {self.decision}>"
