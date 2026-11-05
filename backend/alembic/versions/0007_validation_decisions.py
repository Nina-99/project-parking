# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Add the operator validation queue and link detections to their vehicle.

Revision ID: 0007_validation_decisions
Revises: 0006_event_direction

Two changes that belong together:

* ``validation_decisions`` records what an operator decided about a gate event.
  The queue is the human half of RF-10: the system classifies, the operator
  confirms or corrects.
* ``detection_events.vehicle_id`` links a detection to the vehicle it produced.
  Until now an unmatched exit had no row that connected the two, which made it
  impossible to ask "which recent events has nobody reviewed yet?".
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_validation_decisions"
down_revision = "0006_event_direction"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "detection_events",
        sa.Column("vehicle_id", sa.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_detection_events_vehicle_id",
        "detection_events",
        "vehicles",
        ["vehicle_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_detection_events_vehicle_id", "detection_events", ["vehicle_id"],
    )

    op.create_table(
        "validation_decisions",
        sa.Column("id", sa.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "vehicle_id", sa.UUID(as_uuid=True),
            sa.ForeignKey("vehicles.id"), nullable=False,
        ),
        sa.Column(
            "session_id", sa.UUID(as_uuid=True),
            sa.ForeignKey("parking_sessions.id"), nullable=True,
        ),
        sa.Column("decision", sa.String(length=20), nullable=False),
        sa.Column("original_plate", sa.String(length=20), nullable=True),
        sa.Column("corrected_plate", sa.String(length=20), nullable=True),
        sa.Column("operator", sa.String(length=50), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
    )
    op.create_index(
        "ix_validation_decisions_vehicle_id",
        "validation_decisions", ["vehicle_id"],
    )
    op.create_index(
        "ix_validation_decisions_decision",
        "validation_decisions", ["decision"],
    )
    op.create_index(
        "ix_validation_decisions_operator",
        "validation_decisions", ["operator"],
    )


def downgrade() -> None:
    op.drop_index("ix_validation_decisions_operator", table_name="validation_decisions")
    op.drop_index("ix_validation_decisions_decision", table_name="validation_decisions")
    op.drop_index("ix_validation_decisions_vehicle_id", table_name="validation_decisions")
    op.drop_table("validation_decisions")

    op.drop_index("ix_detection_events_vehicle_id", table_name="detection_events")
    op.drop_constraint(
        "fk_detection_events_vehicle_id", "detection_events", type_="foreignkey",
    )
    op.drop_column("detection_events", "vehicle_id")
