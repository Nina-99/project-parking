# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Initial schema: vehicles, parking sessions and raw detections.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-23
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

session_status = postgresql.ENUM(
    "active", "completed", "cancelled", name="session_status", create_type=False,
)


def upgrade() -> None:
    session_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "vehicles",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("license_plate", sa.String(length=20), nullable=True),
        sa.Column("license_plate_confidence", sa.Float(), nullable=True),
        sa.Column("vehicle_type", sa.String(length=50), nullable=True),
        sa.Column("vehicle_type_confidence", sa.Float(), nullable=True),
        sa.Column("color", sa.String(length=30), nullable=True),
        sa.Column("color_confidence", sa.Float(), nullable=True),
        sa.Column("brand", sa.String(length=50), nullable=True),
        sa.Column("brand_confidence", sa.Float(), nullable=True),
        sa.Column("entry_image_path", sa.Text(), nullable=True),
        sa.Column("entry_video_url", sa.Text(), nullable=True),
        sa.Column("feature_vector", sa.LargeBinary(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_vehicles_license_plate", "vehicles", ["license_plate"])

    op.create_table(
        "parking_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("vehicle_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entry_gate", sa.String(length=20), nullable=True),
        sa.Column("exit_gate", sa.String(length=20), nullable=True),
        sa.Column(
            "entry_time",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("status", session_status, nullable=False),
        sa.Column("exit_image_path", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["vehicle_id"], ["vehicles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_parking_sessions_vehicle_id", "parking_sessions", ["vehicle_id"])
    op.create_index("ix_parking_sessions_status", "parking_sessions", ["status"])

    op.create_table(
        "detection_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("camera_id", sa.String(length=50), nullable=False),
        sa.Column(
            "frame_timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(length=20), nullable=False),
        sa.Column("object_class", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("crop_image_path", sa.Text(), nullable=True),
        sa.Column("inference_latency_ms", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(length=20), nullable=True),
        sa.Column("feature_vector", sa.LargeBinary(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["parking_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_detection_events_session_id", "detection_events", ["session_id"])
    op.create_index("ix_detection_events_camera_id", "detection_events", ["camera_id"])


def downgrade() -> None:
    op.drop_table("detection_events")
    op.drop_table("parking_sessions")
    op.drop_table("vehicles")
    session_status.drop(op.get_bind(), checkfirst=True)
