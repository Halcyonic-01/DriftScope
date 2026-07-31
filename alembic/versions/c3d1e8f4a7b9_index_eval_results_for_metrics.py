"""index eval_results for metrics and drift queries

Every /metrics scrape (every 15s by default) filters eval_results by
evaluated_at and groups by model_version, as does GET /drift and the
reports endpoint. Before this migration the only index was the
result_id primary key, so all of those did sequential scans over a
table that grows with every eval forever.

Revision ID: c3d1e8f4a7b9
Revises: b7e5a4a9d6c2
Create Date: 2026-07-31 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d1e8f4a7b9"
down_revision: Union[str, None] = "b7e5a4a9d6c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Composite ordered (model_version, evaluated_at): serves the
    # group-by-then-time-filter pattern the metrics/drift queries use,
    # and also the model_version-only lookups in /reports.
    op.create_index(
        "ix_eval_results_model_version_evaluated_at",
        "eval_results",
        ["model_version", "evaluated_at"],
    )
    # Time-only index for the window scans that span all model versions.
    op.create_index(
        "ix_eval_results_evaluated_at",
        "eval_results",
        ["evaluated_at"],
    )
    # /cases/{case_id}/results filters by case_id.
    op.create_index(
        "ix_eval_results_case_id",
        "eval_results",
        ["case_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_eval_results_case_id", table_name="eval_results")
    op.drop_index("ix_eval_results_evaluated_at", table_name="eval_results")
    op.drop_index("ix_eval_results_model_version_evaluated_at", table_name="eval_results")
