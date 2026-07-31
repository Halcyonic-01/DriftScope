"""
JobRun records the outcome of one scheduled job execution.

WHY this exists:
  The in-memory Prometheus counters in app/core/instrumentation.py only
  see LLM calls made inside the API process. scripts/run_monitor.py and
  scripts/run_canary.py run as separate processes — on a laptop or a
  GitHub Actions runner — so their provider calls never touch those
  counters.

  That left a real blind spot: a failed case writes no eval_results row,
  so a run where every case failed (expired model ID, exhausted quota)
  looked exactly like a run that never happened.

  Persisting one row per job execution makes that visible from any
  process, with no extra infrastructure, and keeps /metrics as the single
  place the dashboard reads from.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class JobRun(Base):
    __tablename__ = "job_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # "monitor" or "canary" — kept generic so the canary can record here too.
    job: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(100), nullable=True)

    cases_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cases_succeeded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cases_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # One representative error so the dashboard can say *why* a run failed
    # (quota exhausted vs. retired model) without trawling CI logs.
    last_error: Mapped[str] = mapped_column(Text, nullable=True)

    finished_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )

    def __repr__(self) -> str:
        return (
            f"<JobRun job={self.job} provider={self.provider} "
            f"ok={self.cases_succeeded} failed={self.cases_failed}>"
        )
