"""
app/db/models/golden_case.py

GoldenCase — a single evaluation test case.

Think of this as a "question in an exam". Each golden case stores:
  - The prompt (question we send to the LLM)
  - What topics the response should cover (expected_topics)
  - Safety rules the response must follow (safety_rules)
  - Which model/prompt version this case belongs to (version_tag)
  - What domain it's from — e.g. medical, legal, finance (domain)

We store BEHAVIOURAL contracts, not exact expected strings.
This makes evals robust to LLM non-determinism.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class GoldenCase(Base):
    __tablename__ = "golden_cases"

    # UUID primary key — more scalable than integer IDs
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # The prompt we'll send to the LLM under test
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    # Themes/topics the response should cover (array of strings)
    expected_topics: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=True, default=list
    )

    # Natural-language rules for the LLM judge to check
    safety_rules: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=True, default=list
    )

    # e.g. "v1.2-gemini-flash" — ties this case to a model/prompt version
    version_tag: Mapped[str] = mapped_column(String(50), nullable=True)

    # e.g. "medical", "legal", "finance", "general"
    domain: Mapped[str] = mapped_column(String(50), nullable=True)

    # Auto-set to current UTC time on insert
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<GoldenCase id={self.case_id} domain={self.domain}>"
