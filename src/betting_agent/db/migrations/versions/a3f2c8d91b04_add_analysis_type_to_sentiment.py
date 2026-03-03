"""add_analysis_type_to_sentiment

Revision ID: a3f2c8d91b04
Revises: 6b168fed7e76
Create Date: 2026-03-01

"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "a3f2c8d91b04"
down_revision: Union[str, None] = "6b168fed7e76"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "sentiment",
        sa.Column("analysis_type", sa.String(30), server_default="injury", nullable=True),
    )


def downgrade() -> None:
    op.drop_column("sentiment", "analysis_type")
