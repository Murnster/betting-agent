"""Add ledger fields (actual_bet/actual_odds) and prop fields to picks.

Revision ID: b8c1d2e3f4a5
Revises: e4d5b8b7d9c1
Create Date: 2026-08-18

actual_bet / actual_odds record the bet actually placed at the book when it
differs from the recommendation, so the ledger reflects real money.
line / player / market give props (and spreads/totals) structured storage
instead of parsing pick_side strings.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b8c1d2e3f4a5"
down_revision: Union[str, Sequence[str], None] = "e4d5b8b7d9c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("picks", sa.Column("actual_bet", sa.Float(), nullable=True))
    op.add_column("picks", sa.Column("actual_odds", sa.Integer(), nullable=True))
    op.add_column("picks", sa.Column("line", sa.Float(), nullable=True))
    op.add_column("picks", sa.Column("player", sa.String(100), nullable=True))
    op.add_column("picks", sa.Column("market", sa.String(50), nullable=True))


def downgrade() -> None:
    for col in ("market", "player", "line", "actual_odds", "actual_bet"):
        op.drop_column("picks", col)
