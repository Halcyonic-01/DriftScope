"""
CentroidHistory stores provider canary snapshots.

Each row captures the embedding centroid for one provider run plus the drift
score against the recent rolling centroid.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class CentroidHistory(Base):
    __tablename__ = "centroid_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    centroid: Mapped[list[float]] = mapped_column(ARRAY(Float), nullable=False)
    drift_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
    )

    def __repr__(self) -> str:
        return f"<CentroidHistory provider={self.provider} drift={self.drift_score:.4f}>"
