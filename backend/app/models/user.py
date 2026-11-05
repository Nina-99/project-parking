# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""One row per person who can enter the system.

The role decides what they may do, not who they are:

``admin``     manages users and cameras. Not required to operate the lot.
``operator``  runs the lot: closes sessions, corrects data.
``owner``     reads. Sees occupancy, history and reports, changes nothing.

The password is never stored, only its scrypt hash
(``backend/app/core/security.py``).
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class UserRole(enum.StrEnum):
    """Kept as a plain string in the database, validated here and by a CHECK.

    A Postgres ENUM would need a migration to add a role later; the CHECK
    constraint gives the same guarantee without that cost.
    """

    ADMIN = "admin"
    OPERATOR = "operator"
    OWNER = "owner"


ROLE_VALUES: tuple[str, ...] = tuple(role.value for role in UserRole)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "role IN ('admin', 'operator', 'owner')", name="ck_users_role",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default=UserRole.OPERATOR.value)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.username} role={self.role} active={self.is_active}>"
