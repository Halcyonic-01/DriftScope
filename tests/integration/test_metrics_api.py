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


def _run_one_eval(client, model_version: str) -> None:
    case = client.post(
        "/cases",
        json={
            "prompt": "Explain the side effects of ibuprofen in simple terms.",
            "expected_topics": ["side effects", "dosage"],
            "domain": "medical",
        },
    ).json()
    resp = client.post(
        f"/cases/{case['case_id']}/run",
        json={"model_version": model_version, "provider": "mock"},
    )
    assert resp.status_code == 201


def test_metrics_reports_eval_freshness_and_run_count(client):
    """
    Freshness is what makes a stalled eval job visible. Without it, a
    dashboard that stopped updating looks the same as a healthy one.
    """
    _run_one_eval(client, "freshness-test-v1")

    text = client.get("/metrics").text

    assert re.search(
        r'driftscope_eval_last_run_timestamp_seconds\{model_version="freshness-test-v1"\} \d{10}',
        text,
    )
    assert re.search(
        r'driftscope_eval_runs_total\{model_version="freshness-test-v1"\} 1', text
    )


def test_metrics_distinguishes_insufficient_data_from_no_drift(client):
    """
    drift_detected=0 alone is ambiguous. A single eval is far below the
    10/30 sample floor, so the blind signal must be 1 here.
    """
    _run_one_eval(client, "blind-test-v1")

    text = client.get("/metrics").text

    assert 'driftscope_drift_detected{model_version="blind-test-v1"} 0' in text
    assert 'driftscope_drift_insufficient_data{model_version="blind-test-v1"} 1' in text


def test_metrics_includes_canary_freshness(client, db_session):
    from app.db.models.centroid_history import CentroidHistory

    db_session.add(
        CentroidHistory(provider="freshness-canary", centroid=[0.1, 0.2], drift_score=0.01)
    )
    db_session.flush()

    text = client.get("/metrics").text

    assert 'driftscope_canary_centroid_drift{provider="freshness-canary"}' in text
    assert re.search(
        r'driftscope_canary_last_run_timestamp_seconds\{provider="freshness-canary"\} \d{10}',
        text,
    )


def test_metrics_exposes_job_run_outcomes(client, db_session):
    """
    Scheduled runs happen in another process, so the in-memory LLM
    counters never see them. These DB-derived metrics are what make a
    failing scheduled job visible.
    """
    from app.db.models.job_run import JobRun

    db_session.add(
        JobRun(
            job="monitor",
            provider="metrics-prov",
            model_version="v1",
            cases_total=10,
            cases_succeeded=6,
            cases_failed=4,
            last_error="quota exhausted",
        )
    )
    db_session.flush()

    text = client.get("/metrics").text

    labels = 'job="monitor",provider="metrics-prov"'
    assert f"driftscope_job_cases_failed{{{labels}}} 4" in text
    assert f"driftscope_job_cases_succeeded{{{labels}}} 6" in text
    assert f"driftscope_job_failure_ratio{{{labels}}} 0.400000" in text
    assert re.search(
        r'driftscope_job_last_run_timestamp_seconds\{job="monitor",provider="metrics-prov"\} \d{10}',
        text,
    )


def test_metrics_reports_only_the_latest_job_run(client, db_session):
    from app.db.models.job_run import JobRun
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    db_session.add_all([
        JobRun(job="monitor", provider="latest-prov", cases_total=5,
               cases_succeeded=0, cases_failed=5, finished_at=now - timedelta(hours=2)),
        JobRun(job="monitor", provider="latest-prov", cases_total=5,
               cases_succeeded=5, cases_failed=0, finished_at=now),
    ])
    db_session.flush()

    text = client.get("/metrics").text

    labels = 'job="monitor",provider="latest-prov"'
    assert f"driftscope_job_cases_failed{{{labels}}} 0" in text
    assert f"driftscope_job_cases_succeeded{{{labels}}} 5" in text


def test_metrics_includes_process_http_metrics(client):
    """The in-memory HTTP counters must be merged into the same output."""
    client.get("/health")

    text = client.get("/metrics").text

    assert "driftscope_http_requests_total" in text
    assert 'path="/health"' in text
