# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Record the direction the system decided for each event.

Revision ID: 0006_event_direction
Revises: 0005_authorized_vehicles

The value was already computed and published; it just was not kept. Without it a
measurement of event accuracy has to guess the direction from the camera's role,
which is the input, not the decision being evaluated.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_event_direction"
down_revision = "0005_authorized_vehicles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "detection_events",
        sa.Column("direction", sa.String(length=10), nullable=True),
    )
    op.create_index(
        "ix_detection_events_direction", "detection_events", ["direction"],
    )


def downgrade() -> None:
    op.drop_index("ix_detection_events_direction", table_name="detection_events")
    op.drop_column("detection_events", "direction")
