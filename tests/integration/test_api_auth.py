"""
Integration tests confirming API key auth is actually wired into the real
routers (catches "forgot to add dependencies=[...] to a router" mistakes
that a pure unit test of require_api_key() wouldn't catch).
"""

from __future__ import annotations

import pytest

from app.core.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture()
def api_key_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-secret-key")
    return "test-secret-key"


def test_cases_route_rejects_missing_key(client, api_key_configured):
    response = client.get("/cases")
    assert response.status_code == 401


def test_cases_route_rejects_wrong_key(client, api_key_configured):
    response = client.get("/cases", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_cases_route_accepts_correct_key(client, api_key_configured):
    response = client.get("/cases", headers={"X-API-Key": api_key_configured})
    assert response.status_code == 200


def test_drift_and_reports_routes_are_protected_too(client, api_key_configured):
    assert client.get("/drift/some-model").status_code == 401
    assert client.get("/reports/some-model").status_code == 401


def test_health_and_metrics_stay_open_regardless_of_api_key(client, api_key_configured):
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_cases_route_open_when_no_api_key_configured(client):
    # api_key defaults to "" — auth should be a no-op.
    assert client.get("/cases").status_code == 200
