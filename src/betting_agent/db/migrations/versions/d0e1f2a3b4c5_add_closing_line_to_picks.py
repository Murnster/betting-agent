"""add closing_line to picks

Props capture their own closing price from the per-event endpoint shortly
before kickoff. Books move the NUMBER as well as the price, so the line at
close is stored alongside closing_odds: CLV is only computed when the line
held, and the report counts moves for/against the pick otherwise.

Revision ID: d0e1f2a3b4c5
Revises: c9d2e3f4a5b6
Create Date: 2026-09-05

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d0e1f2a3b4c5"
down_revision: Union[str, Sequence[str], None] = "c9d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("picks", sa.Column("closing_line", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("picks", "closing_line")
