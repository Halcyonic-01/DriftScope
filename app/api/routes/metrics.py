"""
Prometheus metrics endpoint.

Two families are exposed here:

  * Quality metrics, derived from eval_results / centroid_history on each
    scrape. These describe MODEL health.
  * Process metrics (HTTP + LLM counters), collected in memory by
    app/core/instrumentation.py. These describe SERVICE health.

Both are needed. Quality gauges alone can't distinguish "the model is
fine" from "the eval job stopped running" or "the provider is rejecting
every call" -- which is why the freshness and status metrics below exist.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.drift import detect_drift_from_scores
from app.db.models.centroid_history import CentroidHistory
from app.db.models.eval_result import EvalResult
from app.db.session import get_db_session

router = APIRouter(tags=["Metrics"])

# Show the last 24 h for quality/judge gauges so the dashboard reflects
# current model health, not an all-time historical average.
METRICS_WINDOW_HOURS = 24
BASELINE_WINDOW_DAYS = 7


@router.get("/metrics")
def metrics(db: Session = Depends(get_db_session)):
    """
    Return Prometheus text format for quality, drift, freshness, and
    process metrics.
    """
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(hours=METRICS_WINDOW_HOURS)
    baseline_start = now - timedelta(days=BASELINE_WINDOW_DAYS)

    lines = [
        "# HELP driftscope_quality_score Average composite quality score (last 24 h).",
        "# TYPE driftscope_quality_score gauge",
        "# HELP driftscope_drift_detected 1 if drift detected, 0 otherwise.",
        "# TYPE driftscope_drift_detected gauge",
        "# HELP driftscope_drift_insufficient_data 1 when there is too little data to run drift detection.",
        "# TYPE driftscope_drift_insufficient_data gauge",
        "# HELP driftscope_judge_pass_rate Average LLM judge pass rate (last 24 h).",
        "# TYPE driftscope_judge_pass_rate gauge",
        "# HELP driftscope_eval_runs_total Eval results recorded in the last 24 h.",
        "# TYPE driftscope_eval_runs_total gauge",
        "# HELP driftscope_eval_last_run_timestamp_seconds Unix time of the most recent eval result.",
        "# TYPE driftscope_eval_last_run_timestamp_seconds gauge",
        "# HELP driftscope_canary_centroid_drift Latest provider canary centroid drift score.",
        "# TYPE driftscope_canary_centroid_drift gauge",
        "# HELP driftscope_canary_last_run_timestamp_seconds Unix time of the most recent canary snapshot.",
        "# TYPE driftscope_canary_last_run_timestamp_seconds gauge",
    ]

    # ── Quality + drift, in two queries total ────────────────────────────
    #
    # This used to call detect_drift() once per model_version inside the
    # loop, i.e. 1 + 2N queries on every scrape (every 15 s, forever).
    # Instead pull the whole baseline+current window once and bucket the
    # scores in Python.
    score_rows = (
        db.query(
            EvalResult.model_version,
            EvalResult.composite_score,
            EvalResult.judge_score,
            EvalResult.evaluated_at,
        )
        .filter(EvalResult.composite_score.isnot(None))
        .filter(EvalResult.evaluated_at >= baseline_start)
        .all()
    )

    current: dict[str, list[float]] = {}
    baseline: dict[str, list[float]] = {}
    judge_scores: dict[str, list[float]] = {}

    for model_version, composite, judge, evaluated_at in score_rows:
        if evaluated_at >= current_start:
            current.setdefault(model_version, []).append(float(composite))
            if judge is not None:
                judge_scores.setdefault(model_version, []).append(float(judge))
        else:
            baseline.setdefault(model_version, []).append(float(composite))

    for model_version, scores in sorted(current.items()):
        label = _label(model_version)
        lines.append(
            f'driftscope_quality_score{{model_version="{label}"}} '
            f"{sum(scores) / len(scores):.6f}"
        )
        lines.append(f'driftscope_eval_runs_total{{model_version="{label}"}} {len(scores)}')

        drift = detect_drift_from_scores(
            model_version=model_version,
            current_scores=scores,
            baseline_scores=baseline.get(model_version, []),
        )
        lines.append(
            f'driftscope_drift_detected{{model_version="{label}"}} '
            f"{1 if drift.drift_detected else 0}"
        )
        # Without this, drift_detected=0 is ambiguous: it means both
        # "healthy" and "we don't have enough data to know".
        lines.append(
            f'driftscope_drift_insufficient_data{{model_version="{label}"}} '
            f"{1 if drift.status == 'insufficient_data' else 0}"
        )

        if model_version in judge_scores:
            judged = judge_scores[model_version]
            lines.append(
                f'driftscope_judge_pass_rate{{model_version="{label}"}} '
                f"{sum(judged) / len(judged):.6f}"
            )

    # ── Freshness: when did each model version last produce a result? ────
    #
    # Deliberately unbounded by time window, so a model that stopped being
    # evaluated still reports a (stale) timestamp and can be alerted on.
    freshness_rows = (
        db.query(EvalResult.model_version, func.max(EvalResult.evaluated_at))
        .group_by(EvalResult.model_version)
        .all()
    )
    for model_version, last_seen in freshness_rows:
        if last_seen is not None:
            lines.append(
                f'driftscope_eval_last_run_timestamp_seconds{{model_version="{_label(model_version)}"}} '
                f"{_epoch(last_seen):.0f}"
            )

    # ── Canary ───────────────────────────────────────────────────────────
    latest_canaries = (
        db.query(
            CentroidHistory.provider,
            func.max(CentroidHistory.recorded_at).label("latest_recorded_at"),
        )
        .group_by(CentroidHistory.provider)
        .subquery()
    )
    canary_rows = (
        db.query(
            CentroidHistory.provider,
            CentroidHistory.drift_score,
            CentroidHistory.recorded_at,
        )
        .join(
            latest_canaries,
            (CentroidHistory.provider == latest_canaries.c.provider)
            & (CentroidHistory.recorded_at == latest_canaries.c.latest_recorded_at),
        )
        .all()
    )

    for provider, drift_score, recorded_at in canary_rows:
        label = _label(provider)
        lines.append(
            f'driftscope_canary_centroid_drift{{provider="{label}"}} {float(drift_score):.6f}'
        )
        # A canary that died looks identical to a healthy one without this.
        lines.append(
            f'driftscope_canary_last_run_timestamp_seconds{{provider="{label}"}} '
            f"{_epoch(recorded_at):.0f}"
        )

    body = "\n".join(lines) + "\n"
    # Append in-memory HTTP/LLM counters from the default registry.
    body += generate_latest().decode("utf-8")

    return Response(body, media_type=CONTENT_TYPE_LATEST)


def _epoch(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
