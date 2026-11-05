# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Drop fields nothing produces.

``feature_vector`` was measured and rejected as an identity signal; ``brand``
has no classifier; ``entry_video_url`` and ``exit_image_path`` were never
written. Keeping them forces every reader to wonder which ones are real.

Revision ID: 0003_lean_schema
Revises: 0002_session_close_audit
Create Date: 2026-10-23
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_lean_schema"
down_revision = "0002_session_close_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("vehicles", "feature_vector")
    op.drop_column("vehicles", "brand")
    op.drop_column("vehicles", "brand_confidence")
    op.drop_column("vehicles", "entry_video_url")
    op.drop_column("detection_events", "feature_vector")
    op.drop_column("parking_sessions", "exit_image_path")


def downgrade() -> None:
    op.add_column("parking_sessions", sa.Column("exit_image_path", sa.String(), nullable=True))
    op.add_column(
        "detection_events",
        sa.Column("feature_vector", sa.LargeBinary(), nullable=True),
    )
    op.add_column("vehicles", sa.Column("entry_video_url", sa.Text(), nullable=True))
    op.add_column(
        "vehicles",
        sa.Column("brand_confidence", sa.Float(), nullable=True),
    )
    op.add_column("vehicles", sa.Column("brand", sa.String(length=50), nullable=True))
    op.add_column("vehicles", sa.Column("feature_vector", sa.LargeBinary(), nullable=True))
