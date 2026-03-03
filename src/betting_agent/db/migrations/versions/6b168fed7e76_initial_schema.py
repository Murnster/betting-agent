"""initial_schema

Revision ID: 6b168fed7e76
Revises:
Create Date: 2026-02-28

"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "6b168fed7e76"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "games",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("sport", sa.String(length=10), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("week", sa.Integer(), nullable=True),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("home_team", sa.String(length=50), nullable=False),
        sa.Column("away_team", sa.String(length=50), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("neutral_site", sa.Boolean(), nullable=True, server_default="false"),
        sa.Column("weather_desc", sa.String(length=100), nullable=True),
        sa.Column("temperature_f", sa.Float(), nullable=True),
        sa.Column("wind_mph", sa.Float(), nullable=True),
        sa.Column("is_playoff", sa.Boolean(), nullable=True, server_default="false"),
        sa.Column("status", sa.String(length=20), nullable=True, server_default="scheduled"),
        sa.Column("external_id", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )

    op.create_table(
        "odds",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("bookmaker", sa.String(length=50), nullable=True),
        sa.Column("bet_type", sa.String(length=20), nullable=True),
        sa.Column("market_key", sa.String(length=100), nullable=True),
        sa.Column("description", sa.String(length=200), nullable=True),
        sa.Column("home_price", sa.Integer(), nullable=True),
        sa.Column("away_price", sa.Integer(), nullable=True),
        sa.Column("spread_home", sa.Float(), nullable=True),
        sa.Column("total_line", sa.Float(), nullable=True),
        sa.Column("fetched_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("is_closing", sa.Boolean(), nullable=True, server_default="false"),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "sentiment",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("team", sa.String(length=50), nullable=True),
        sa.Column("source", sa.String(length=100), nullable=True),
        sa.Column("sentiment_score", sa.Float(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("model_used", sa.String(length=50), nullable=True),
        sa.Column("analyzed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "picks",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("sport", sa.String(length=10), nullable=False),
        sa.Column("pick_date", sa.Date(), nullable=False),
        sa.Column("bet_type", sa.String(length=20), nullable=False),
        sa.Column("pick_side", sa.String(length=100), nullable=False),
        sa.Column("model_prob", sa.Float(), nullable=False),
        sa.Column("implied_prob", sa.Float(), nullable=False),
        sa.Column("edge", sa.Float(), nullable=False),
        sa.Column("odds", sa.Integer(), nullable=False),
        sa.Column("kelly_fraction", sa.Float(), nullable=True),
        sa.Column("recommended_bet", sa.Float(), nullable=True),
        sa.Column("bankroll_at_pick", sa.Float(), nullable=True),
        sa.Column("result", sa.String(length=10), nullable=True),
        sa.Column("closing_odds", sa.Integer(), nullable=True),
        sa.Column("clv", sa.Float(), nullable=True),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("graded_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("picks")
    op.drop_table("sentiment")
    op.drop_table("odds")
    op.drop_table("games")
