"""Add the audit fields for a session closed by an operator.

Revision ID: 0002_session_close_audit
Revises: 0001_initial
Create Date: 2026-10-23
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_session_close_audit"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "parking_sessions",
        sa.Column("close_reason", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "parking_sessions",
        sa.Column("closed_by", sa.String(length=50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("parking_sessions", "closed_by")
    op.drop_column("parking_sessions", "close_reason")
