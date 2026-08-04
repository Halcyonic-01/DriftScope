"""add case_count to centroid_history

A centroid is the mean embedding of its prompt set, so snapshots built
from different numbers of prompts are not comparable. Mixing a 5-case and
a 20-case snapshot in the rolling baseline produced drift scores an order
of magnitude above the 0.05 alert threshold that reflected the changed
input, not the model.

Existing rows are backfilled to NULL rather than guessed: their prompt
set is unknown, and NULL excludes them from same-size comparisons, which
is the safe behaviour.

Revision ID: e5b3c9d2f8a1
Revises: d4a2f6b8c1e3
Create Date: 2026-08-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "e5b3c9d2f8a1"
down_revision: Union[str, None] = "d4a2f6b8c1e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "centroid_history",
        sa.Column("case_count", sa.Integer(), nullable=True),
    )
    op.create_index(
        op.f("ix_centroid_history_case_count"),
        "centroid_history",
        ["case_count"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_centroid_history_case_count"), table_name="centroid_history")
    op.drop_column("centroid_history", "case_count")
