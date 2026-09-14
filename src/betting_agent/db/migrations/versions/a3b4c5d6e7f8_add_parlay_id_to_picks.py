"""add parlay_id to picks

Revision ID: a3b4c5d6e7f8
Revises: f2a3b4c5d6e7
Create Date: 2026-09-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a3b4c5d6e7f8"
down_revision = "f2a3b4c5d6e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A parlay is a Pick row (bet_type "parlay", strategy "parlay") whose legs
    # are ordinary Pick rows (strategy "parlay_leg", stake 0) pointing back at
    # it. Legs grade through the normal graders; the parent settles as the AND
    # of its legs (accounting/parlays.py).
    op.add_column("picks", sa.Column("parlay_id", sa.Integer(), nullable=True))
    op.create_foreign_key("fk_picks_parlay_id", "picks", "picks", ["parlay_id"], ["id"])
    op.create_index("ix_picks_parlay_id", "picks", ["parlay_id"])


def downgrade() -> None:
    op.drop_index("ix_picks_parlay_id", table_name="picks")
    op.drop_constraint("fk_picks_parlay_id", "picks", type_="foreignkey")
    op.drop_column("picks", "parlay_id")
