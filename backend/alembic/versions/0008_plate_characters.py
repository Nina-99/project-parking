# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Keep the per-character read next to the plate.

Revision ID: 0008_plate_characters
Revises: 0007_validation_decisions

The OCR already computes a probability for every emitted character; it was
discarded after the mean was taken. Storing the distribution lets the operator see
which character is weak and correct one position instead of retyping the plate,
which is the recommendation the results chapter makes.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008_plate_characters"
down_revision = "0007_validation_decisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "vehicles",
        sa.Column("plate_characters", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("vehicles", "plate_characters")
