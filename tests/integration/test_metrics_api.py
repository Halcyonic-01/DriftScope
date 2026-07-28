"""
Integration tests for GET /metrics (Prometheus exposition format).
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.integration


def test_metrics_returns_prometheus_text_with_no_data(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# HELP driftscope_quality_score" in response.text
    assert "# TYPE driftscope_quality_score gauge" in response.text
    assert "# HELP driftscope_canary_centroid_drift" in response.text


def test_metrics_includes_quality_score_after_eval_run(client):
    case = client.post(
        "/cases",
        json={
            "prompt": "Explain the side effects of ibuprofen in simple terms.",
            "expected_topics": ["side effects", "dosage"],
            "domain": "medical",
        },
    ).json()
    client.post(
        f"/cases/{case['case_id']}/run",
        json={"model_version": "metrics-test-v1", "provider": "mock"},
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    pattern = r'driftscope_quality_score\{model_version="metrics-test-v1"\} \d+\.\d+'
    assert re.search(pattern, response.text)
    drift_pattern = r'driftscope_drift_detected\{model_version="metrics-test-v1"\} [01]'
    assert re.search(drift_pattern, response.text)
