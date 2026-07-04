"""add centroid_history table

Revision ID: b7e5a4a9d6c2
Revises: 958af7f2871c
Create Date: 2026-07-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b7e5a4a9d6c2"
down_revision: Union[str, None] = "958af7f2871c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "centroid_history",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("centroid", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("drift_score", sa.Float(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_centroid_history_provider"),
        "centroid_history",
        ["provider"],
        unique=False,
    )
    op.create_index(
        op.f("ix_centroid_history_recorded_at"),
        "centroid_history",
        ["recorded_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_centroid_history_recorded_at"), table_name="centroid_history")
    op.drop_index(op.f("ix_centroid_history_provider"), table_name="centroid_history")
    op.drop_table("centroid_history")
