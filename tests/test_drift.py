"""
Tests for Phase 3 drift detection.
"""

from __future__ import annotations

from app.core.drift import detect_drift_from_scores


def test_detect_drift_returns_insufficient_data_for_small_windows():
    result = detect_drift_from_scores(
        model_version="test-model",
        current_scores=[0.7, 0.72],
        baseline_scores=[0.8, 0.81],
    )

    assert result.status == "insufficient_data"
    assert result.drift_detected is False
    assert result.p_value is None
    assert result.current_count == 2
    assert result.baseline_count == 2


def test_detect_drift_flags_large_quality_drop():
    baseline = [0.88, 0.89, 0.9, 0.91, 0.92] * 8
    current = [0.58, 0.6, 0.61, 0.62, 0.63] * 3

    result = detect_drift_from_scores(
        model_version="test-model",
        current_scores=current,
        baseline_scores=baseline,
    )

    assert result.status == "ok"
    assert result.drift_detected is True
    assert result.p_value < 0.05
    assert result.effect_size > 0.1
    assert result.current_mean < result.baseline_mean


def test_detect_drift_does_not_flag_equivalent_distributions():
    baseline = [0.78, 0.8, 0.82, 0.81, 0.79] * 8
    current = [0.79, 0.8, 0.81, 0.82, 0.78] * 3

    result = detect_drift_from_scores(
        model_version="test-model",
        current_scores=current,
        baseline_scores=baseline,
    )

    assert result.status == "ok"
    assert result.drift_detected is False
