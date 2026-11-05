# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Authorized vehicles registry (padrón de vehículos habilitados).

Allows auditing access permissions against a central registry (RF-10).

Revision ID: 0005_authorized_vehicles
Revises: 0004_users_and_cameras
Create Date: 2026-10-29
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_authorized_vehicles"
down_revision = "0004_users_and_cameras"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "authorized_vehicles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("license_plate", sa.String(length=20), nullable=False),
        sa.Column("owner_name", sa.String(length=120), nullable=True),
        sa.Column("vehicle_type", sa.String(length=50), nullable=True),
        sa.Column("color", sa.String(length=30), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_authorized_vehicles_plate",
        "authorized_vehicles",
        ["license_plate"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_authorized_vehicles_plate", table_name="authorized_vehicles")
    op.drop_table("authorized_vehicles")
