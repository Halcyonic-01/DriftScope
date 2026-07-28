"""
Integration tests for GET /reports/{model_version}.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _create_and_run(client, model_version: str, provider: str = "mock") -> None:
    case = client.post(
        "/cases",
        json={
            "prompt": "Explain the side effects of ibuprofen in simple terms.",
            "expected_topics": ["side effects", "dosage"],
            "safety_rules": [],
            "domain": "medical",
        },
    ).json()
    run = client.post(
        f"/cases/{case['case_id']}/run",
        json={"model_version": model_version, "provider": provider},
    )
    assert run.status_code == 201


def test_report_aggregates_runs_for_model_version(client):
    model_version = "report-test-v1"
    _create_and_run(client, model_version)
    _create_and_run(client, model_version)

    response = client.get(f"/reports/{model_version}")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == model_version
    assert body["total_runs"] == 2
    assert body["avg_composite_score"] is not None
    assert 0.0 <= body["avg_composite_score"] <= 1.0
    assert 0.0 <= body["judge_invocation_rate"] <= 1.0
    assert body["evaluated_from"] is not None
    assert body["evaluated_to"] is not None


def test_report_returns_zeroed_state_for_unseen_model_version(client):
    response = client.get("/reports/never-run-before-model")

    assert response.status_code == 200
    body = response.json()
    assert body["total_runs"] == 0
    assert body["avg_composite_score"] is None
    assert body["avg_cosine_score"] is None
    assert body["judge_pass_rate"] is None
    assert body["judge_invocation_rate"] == 0.0
    assert body["evaluated_from"] is None
    assert body["evaluated_to"] is None


def test_report_only_counts_matching_model_version(client):
    _create_and_run(client, "report-test-a")
    _create_and_run(client, "report-test-b")

    response = client.get("/reports/report-test-a")

    assert response.status_code == 200
    assert response.json()["total_runs"] == 1
