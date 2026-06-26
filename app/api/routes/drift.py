"""
Phase 3 drift detection endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.schemas import DriftDetectionResponse
from app.core.drift import detect_drift
from app.db.session import get_db_session

router = APIRouter(prefix="/drift", tags=["Drift"])


@router.get("/{model_version}", response_model=DriftDetectionResponse)
def get_drift_status(
    model_version: str,
    current_hours: int = Query(default=24, ge=1, le=168),
    baseline_days: int = Query(default=7, ge=1, le=90),
    db: Session = Depends(get_db_session),
):
    """
    Compare the latest score window against the older baseline window.
    """
    result = detect_drift(
        db=db,
        model_version=model_version,
        current_hours=current_hours,
        baseline_days=baseline_days,
    )
    return DriftDetectionResponse(**result.__dict__)
