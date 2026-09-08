"""add strategy to picks

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-09-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f2a3b4c5d6e7"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Which paper book a pick belongs to. NULL = the default strategy for its
    # bet_type/market (receiving props, game leans, TD scorers); "ladder" =
    # the ladder-hits section, tracked against its own bankroll.
    op.add_column("picks", sa.Column("strategy", sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column("picks", "strategy")
