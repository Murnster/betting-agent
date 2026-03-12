"""add_agent_validations_table

Revision ID: e4d5b8b7d9c1
Revises: 7666f926b7d0
Create Date: 2026-03-12

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e4d5b8b7d9c1"
down_revision: Union[str, Sequence[str], None] = "7666f926b7d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_validations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("game_id", sa.Integer(), sa.ForeignKey("games.id"), nullable=True),
        sa.Column("external_id", sa.String(length=100), nullable=True),
        sa.Column("pick_date", sa.Date(), nullable=False),
        sa.Column("sport", sa.String(length=10), nullable=False),
        sa.Column("bet_type", sa.String(length=20), nullable=False),
        sa.Column("pick_side", sa.String(length=100), nullable=False),
        sa.Column("verdict", sa.String(length=12), nullable=False),
        sa.Column("original_edge", sa.Float(), nullable=True),
        sa.Column("adjusted_edge", sa.Float(), nullable=True),
        sa.Column("reasons_json", sa.JSON(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("validated_at", sa.TIMESTAMP(), server_default=sa.text("CURRENT_TIMESTAMP")),
    )


def downgrade() -> None:
    op.drop_table("agent_validations")
