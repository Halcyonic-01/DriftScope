"""
Prometheus metrics endpoint.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.drift import detect_drift
from app.db.models.eval_result import EvalResult
from app.db.session import get_db_session

router = APIRouter(tags=["Metrics"])

# Show the last 24 h for quality/judge gauges so the dashboard reflects
# current model health, not an all-time historical average.
METRICS_WINDOW_HOURS = 24


@router.get("/metrics")
def metrics(db: Session = Depends(get_db_session)):
    """
    Return Prometheus text format for quality, drift, and judge pass metrics.

    quality_score and judge_pass_rate are averaged over the last 24 h so
    that the Grafana gauge reacts to recent changes, not diluted by old data.
    """
    lines = [
        "# HELP driftscope_quality_score Average composite quality score (last 24 h).",
        "# TYPE driftscope_quality_score gauge",
        "# HELP driftscope_drift_detected 1 if drift detected, 0 otherwise.",
        "# TYPE driftscope_drift_detected gauge",
        "# HELP driftscope_judge_pass_rate Average LLM judge pass rate (last 24 h).",
        "# TYPE driftscope_judge_pass_rate gauge",
    ]

    since = datetime.now(timezone.utc) - timedelta(hours=METRICS_WINDOW_HOURS)

    rows = (
        db.query(
            EvalResult.model_version,
            func.avg(EvalResult.composite_score),
            func.avg(EvalResult.judge_score),
        )
        .filter(EvalResult.composite_score.isnot(None))
        .filter(EvalResult.evaluated_at >= since)
        .group_by(EvalResult.model_version)
        .all()
    )

    for model_version, avg_composite, judge_pass_rate in rows:
        label = _label(model_version)
        lines.append(f'driftscope_quality_score{{model_version="{label}"}} {float(avg_composite):.6f}')

        drift = detect_drift(db, model_version)
        lines.append(
            f'driftscope_drift_detected{{model_version="{label}"}} '
            f"{1 if drift.drift_detected else 0}"
        )

        if judge_pass_rate is not None:
            lines.append(
                f'driftscope_judge_pass_rate{{model_version="{label}"}} '
                f"{float(judge_pass_rate):.6f}"
            )

    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
