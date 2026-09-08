"""add on_card to picks

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
Create Date: 2026-09-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "e1f2a3b4c5d6"
down_revision = "d0e1f2a3b4c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("picks", sa.Column("on_card", sa.Boolean(), nullable=False,
                                     server_default=sa.text("false")))


def downgrade() -> None:
    op.drop_column("picks", "on_card")
