"""add job_runs table

Records one row per scheduled job execution so failures are visible from
any process. Previously a failed case wrote no eval_results row, making a
fully-failed run indistinguishable from a run that never happened.

Revision ID: d4a2f6b8c1e3
Revises: c3d1e8f4a7b9
Create Date: 2026-07-31 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "d4a2f6b8c1e3"
down_revision: Union[str, None] = "c3d1e8f4a7b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "job_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("job", sa.String(length=50), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=True),
        sa.Column("cases_total", sa.Integer(), nullable=False),
        sa.Column("cases_succeeded", sa.Integer(), nullable=False),
        sa.Column("cases_failed", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_job_runs_job"), "job_runs", ["job"])
    op.create_index(op.f("ix_job_runs_provider"), "job_runs", ["provider"])
    op.create_index(op.f("ix_job_runs_finished_at"), "job_runs", ["finished_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_job_runs_finished_at"), table_name="job_runs")
    op.drop_index(op.f("ix_job_runs_provider"), table_name="job_runs")
    op.drop_index(op.f("ix_job_runs_job"), table_name="job_runs")
    op.drop_table("job_runs")
