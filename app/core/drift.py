"""
Phase 3 statistical drift detection.

DriftScope compares score distributions, not individual scores. The latest
window is tested against a baseline window using Mann-Whitney U plus Cohen's d.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
from scipy.stats import mannwhitneyu
from sqlalchemy.orm import Session

from app.db.models.eval_result import EvalResult

MIN_CURRENT_SAMPLES = 10
MIN_BASELINE_SAMPLES = 30
DEFAULT_CURRENT_HOURS = 24
DEFAULT_BASELINE_DAYS = 7
DEFAULT_P_VALUE_THRESHOLD = 0.05
DEFAULT_EFFECT_SIZE_THRESHOLD = 0.1


@dataclass(frozen=True)
class DriftResult:
    model_version: str
    status: str
    drift_detected: bool
    p_value: float | None
    effect_size: float | None
    current_mean: float | None
    baseline_mean: float | None
    current_count: int
    baseline_count: int


def detect_drift(
    db: Session,
    model_version: str,
    current_hours: int = DEFAULT_CURRENT_HOURS,
    baseline_days: int = DEFAULT_BASELINE_DAYS,
) -> DriftResult:
    """
    Fetch rolling score windows and run distribution-based drift detection.
    """
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(hours=current_hours)
    baseline_start = now - timedelta(days=baseline_days)

    current_scores = get_scores(
        db=db,
        model_version=model_version,
        start_at=current_start,
    )
    baseline_scores = get_scores(
        db=db,
        model_version=model_version,
        start_at=baseline_start,
        end_before=current_start,
    )

    return detect_drift_from_scores(
        model_version=model_version,
        current_scores=current_scores,
        baseline_scores=baseline_scores,
    )


def get_scores(
    db: Session,
    model_version: str,
    start_at: datetime,
    end_before: datetime | None = None,
) -> list[float]:
    """
    Return non-null composite scores for a model version in a time window.
    """
    query = (
        db.query(EvalResult.composite_score)
        .filter(EvalResult.model_version == model_version)
        .filter(EvalResult.composite_score.isnot(None))
        .filter(EvalResult.evaluated_at >= start_at)
    )

    if end_before is not None:
        query = query.filter(EvalResult.evaluated_at < end_before)

    return [float(row[0]) for row in query.all()]


def detect_drift_from_scores(
    model_version: str,
    current_scores: Sequence[float],
    baseline_scores: Sequence[float],
    min_current_samples: int = MIN_CURRENT_SAMPLES,
    min_baseline_samples: int = MIN_BASELINE_SAMPLES,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
    effect_size_threshold: float = DEFAULT_EFFECT_SIZE_THRESHOLD,
) -> DriftResult:
    """
    Compare current scores against baseline scores.

    `alternative="less"` means DriftScope is specifically looking for the
    current distribution getting worse than the baseline distribution.
    """
    current = _clean_scores(current_scores)
    baseline = _clean_scores(baseline_scores)

    if len(current) < min_current_samples or len(baseline) < min_baseline_samples:
        return DriftResult(
            model_version=model_version,
            status="insufficient_data",
            drift_detected=False,
            p_value=None,
            effect_size=None,
            current_mean=_mean_or_none(current),
            baseline_mean=_mean_or_none(baseline),
            current_count=len(current),
            baseline_count=len(baseline),
        )

    _, p_value = mannwhitneyu(current, baseline, alternative="less")
    effect_size = _cohens_d_drop(current, baseline)
    drift_detected = (
        p_value < p_value_threshold
        and effect_size > effect_size_threshold
        and float(np.mean(current)) < float(np.mean(baseline))
    )

    return DriftResult(
        model_version=model_version,
        status="ok",
        drift_detected=bool(drift_detected),
        p_value=round(float(p_value), 4),
        effect_size=round(float(effect_size), 3),
        current_mean=round(float(np.mean(current)), 3),
        baseline_mean=round(float(np.mean(baseline)), 3),
        current_count=len(current),
        baseline_count=len(baseline),
    )


def _clean_scores(scores: Sequence[float]) -> list[float]:
    return [
        float(score)
        for score in scores
        if score is not None and np.isfinite(float(score))
    ]


def _mean_or_none(scores: Sequence[float]) -> float | None:
    if not scores:
        return None
    return round(float(np.mean(scores)), 3)


def _cohens_d_drop(current: Sequence[float], baseline: Sequence[float]) -> float:
    current_std = float(np.std(current, ddof=1))
    baseline_std = float(np.std(baseline, ddof=1))
    pooled_std = np.sqrt((current_std**2 + baseline_std**2) / 2)
    return (float(np.mean(baseline)) - float(np.mean(current))) / (pooled_std + 1e-9)
