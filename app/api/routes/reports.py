"""
Aggregated Phase 2 reporting endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.schemas import ModelReportResponse
from app.db.models.eval_result import EvalResult
from app.db.session import get_db_session

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/{model_version}", response_model=ModelReportResponse)
def get_model_report(
    model_version: str,
    db: Session = Depends(get_db_session),
):
    """
    Return aggregate quality statistics for all runs of a model version.
    """
    base_query = db.query(EvalResult).filter(EvalResult.model_version == model_version)

    total_runs = base_query.count()
    judge_invocations = (
        base_query
        .filter(EvalResult.judge_score.isnot(None))
        .count()
    )

    aggregates = (
        db.query(
            func.avg(EvalResult.composite_score),
            func.avg(EvalResult.cosine_score),
            func.avg(EvalResult.judge_score),
            func.min(EvalResult.evaluated_at),
            func.max(EvalResult.evaluated_at),
        )
        .filter(EvalResult.model_version == model_version)
        .one()
    )

    avg_composite, avg_cosine, judge_pass_rate, evaluated_from, evaluated_to = aggregates

    return ModelReportResponse(
        model_version=model_version,
        total_runs=total_runs,
        avg_composite_score=_round_or_none(avg_composite),
        avg_cosine_score=_round_or_none(avg_cosine),
        judge_pass_rate=_round_or_none(judge_pass_rate),
        judge_invocation_rate=(
            round(judge_invocations / total_runs, 3) if total_runs else 0.0
        ),
        evaluated_from=evaluated_from,
        evaluated_to=evaluated_to,
    )


def _round_or_none(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None else None
