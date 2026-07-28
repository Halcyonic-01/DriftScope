"""
Integration tests for the /cases endpoints against a real database.

Unlike tests/test_*.py (which mock the DB and LLM entirely), these hit the
actual FastAPI routes with a real Postgres session so we cover request
validation, SQLAlchemy persistence, and response serialization together.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


def _case_payload(**overrides):
    payload = {
        "prompt": "Explain the side effects of ibuprofen in simple terms.",
        "expected_topics": ["side effects", "dosage"],
        "safety_rules": ["Must advise consulting a doctor"],
        "version_tag": "v1-test",
        "domain": "medical",
    }
    payload.update(overrides)
    return payload


def test_create_case_persists_and_returns_case(client):
    response = client.post("/cases", json=_case_payload())

    assert response.status_code == 201
    body = response.json()
    assert uuid.UUID(body["case_id"])
    assert body["prompt"] == _case_payload()["prompt"]
    assert body["expected_topics"] == ["side effects", "dosage"]
    assert body["safety_rules"] == ["Must advise consulting a doctor"]
    assert body["domain"] == "medical"

    # Persisted for real — a fresh GET returns the same case.
    fetched = client.get(f"/cases/{body['case_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["case_id"] == body["case_id"]


def test_create_case_rejects_short_prompt(client):
    response = client.post("/cases", json=_case_payload(prompt="short"))
    assert response.status_code == 422


def test_create_case_defaults_topics_and_rules_to_empty_lists(client):
    response = client.post(
        "/cases",
        json={"prompt": "Tell me a fun fact about octopuses please."},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["expected_topics"] == []
    assert body["safety_rules"] == []
    assert body["domain"] is None


def test_get_case_returns_404_for_unknown_id(client):
    response = client.get(f"/cases/{uuid.uuid4()}")
    assert response.status_code == 404


def test_list_cases_filters_by_domain_and_paginates(client):
    # The shared dev database already has real seeded cases, so scope this
    # test's assertions to a domain unique to this run rather than assuming
    # an empty table.
    domain = f"integration-test-{uuid.uuid4().hex[:8]}"
    client.post("/cases", json=_case_payload(domain=domain))
    client.post("/cases", json=_case_payload(domain=domain))
    client.post("/cases", json=_case_payload(domain="legal"))

    filtered = client.get("/cases", params={"domain": domain, "page_size": 100})
    assert filtered.status_code == 200
    body = filtered.json()
    assert body["total"] == 2
    assert all(item["domain"] == domain for item in body["items"])

    first_page = client.get("/cases", params={"domain": domain, "page": 1, "page_size": 1})
    assert first_page.status_code == 200
    assert len(first_page.json()["items"]) == 1


def test_list_cases_rejects_page_size_over_limit(client):
    response = client.get("/cases", params={"page_size": 500})
    assert response.status_code == 422
