"""Split NHL hits (body checks) from MLB hits (batting)

The Game model declared home_hits/away_hits twice — once in the NHL block,
once in the MLB block — and Python silently kept only the MLB pair, so both
sports would have written into one column. The NHL stats get their own
columns; games.home_hits/away_hits stay as the MLB batting stats.

Revision ID: c9d2e3f4a5b6
Revises: b8c1d2e3f4a5
Create Date: 2026-08-19
"""

from alembic import op
import sqlalchemy as sa

revision = "c9d2e3f4a5b6"
down_revision = "b8c1d2e3f4a5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("games", sa.Column("home_nhl_hits", sa.Integer(), nullable=True))
    op.add_column("games", sa.Column("away_nhl_hits", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("games", "away_nhl_hits")
    op.drop_column("games", "home_nhl_hits")
