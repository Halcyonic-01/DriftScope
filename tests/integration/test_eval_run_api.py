"""
Integration tests for POST /cases/{case_id}/run and GET /cases/{case_id}/results.

Uses provider="mock" (a real, first-class provider shipped by the app —
see app/core/llm/mock_client.py) so these tests exercise the genuine
eval_service pipeline (embeddings, cost guard, judge, composite scoring)
without any network calls or API costs.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


def _create_case(client, **overrides) -> str:
    payload = {
        "prompt": "Explain the side effects of ibuprofen in simple terms.",
        "expected_topics": ["side effects", "dosage"],
        "safety_rules": ["Must advise consulting a doctor"],
        "domain": "medical",
    }
    payload.update(overrides)
    response = client.post("/cases", json=payload)
    assert response.status_code == 201
    return response.json()["case_id"]


def test_run_eval_with_mock_provider_persists_scored_result(client):
    case_id = _create_case(client)

    response = client.post(
        f"/cases/{case_id}/run",
        json={"model_version": "v1-test", "provider": "mock"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["case_id"] == case_id
    assert body["model_version"] == "v1-test"
    assert body["provider"] == "mock"
    assert body["response_text"]
    assert 0.0 <= body["cosine_score"] <= 1.0
    assert 0.0 <= body["composite_score"] <= 1.0


def test_run_eval_without_contract_skips_scoring(client):
    case_id = _create_case(client, expected_topics=[], safety_rules=[])

    response = client.post(
        f"/cases/{case_id}/run",
        json={"model_version": "v1-test", "provider": "mock"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["cosine_score"] is None
    assert body["judge_score"] is None
    assert body["composite_score"] is None


def test_run_eval_unknown_provider_returns_400(client):
    case_id = _create_case(client)

    response = client.post(
        f"/cases/{case_id}/run",
        json={"model_version": "v1-test", "provider": "not-a-real-provider"},
    )

    assert response.status_code == 400


def test_run_eval_missing_case_returns_404(client):
    response = client.post(
        f"/cases/{uuid.uuid4()}/run",
        json={"model_version": "v1-test", "provider": "mock"},
    )
    assert response.status_code == 404


def test_get_case_results_returns_full_history(client):
    case_id = _create_case(client)

    client.post(f"/cases/{case_id}/run", json={"model_version": "v1-test", "provider": "mock"})
    client.post(f"/cases/{case_id}/run", json={"model_version": "v1-test", "provider": "mock"})

    response = client.get(f"/cases/{case_id}/results")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {item["case_id"] for item in body["items"]} == {case_id}


def test_get_case_results_returns_404_for_unknown_case(client):
    response = client.get(f"/cases/{uuid.uuid4()}/results")
    assert response.status_code == 404
