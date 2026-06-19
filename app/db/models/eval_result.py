"""
app/db/models/eval_result.py

EvalResult — stores the outcome of running a golden case against an LLM.

Think of this as an "answer sheet". Every time we run an eval:
  1. We send the prompt to the LLM
  2. We embed its response and compute cosine_score
  3. Optionally run LLM-as-judge and get judge_score
  4. Combine them into composite_score
  5. Store ALL of that here

This table is the heart of DriftScope's drift detection —
the rolling window queries in Phase 3 read from here.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class EvalResult(Base):
    __tablename__ = "eval_results"

    result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Which golden case this result is for
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("golden_cases.case_id"),
        nullable=False,
    )

    # e.g. "v1.2-gemini-flash" — used for grouping in drift detection
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)

    # The actual text the LLM returned
    response_text: Mapped[str] = mapped_column(Text, nullable=True)

    # Signal 1: cosine similarity between response embedding and reference
    # Range: 0.0 (completely different) → 1.0 (identical meaning)
    cosine_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Signal 2: LLM-as-judge pass/fail (1.0 = pass, 0.0 = fail, None = skipped)
    judge_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Signal 3: weighted combo of cosine + judge
    # Formula: (0.6 * cosine_score) + (0.4 * judge_score)
    composite_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Why the judge passed/failed — useful for debugging regressions
    judge_reason: Mapped[str] = mapped_column(Text, nullable=True)

    # Which provider generated this response: "gemini", "ollama", "local"
    provider: Mapped[str] = mapped_column(String(50), nullable=True)

    # Auto-set to current UTC time — critical for rolling window queries
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship back to the golden case (optional but useful)
    case = relationship("GoldenCase", backref="results", lazy="select")

    def __repr__(self) -> str:
        return (
            f"<EvalResult id={self.result_id} "
            f"composite={self.composite_score:.3f}>"
        )
