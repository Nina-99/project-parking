# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Users with roles, and the camera registry.

Two things the system assumed but never stored:

* who may enter (a single username/password in configuration is one account for
  everyone, and an administrator of a parking lot is a person, not a constant);
* which cameras should be reporting (heartbeats live in Redis and expire, so a
  camera that never starts is invisible instead of offline).

Revision ID: 0004_users_and_cameras
Revises: 0003_lean_schema
Create Date: 2026-10-26
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_users_and_cameras"
down_revision = "0003_lean_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False,
                  server_default="operator"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("role IN ('admin', 'operator', 'owner')", name="ck_users_role"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "cameras",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("gate", sa.String(length=20), nullable=False, server_default="entry"),
        sa.Column("label", sa.String(length=120), nullable=True),
        sa.Column("rtsp_url", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_cameras_camera_id", "cameras", ["camera_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_cameras_camera_id", table_name="cameras")
    op.drop_table("cameras")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
