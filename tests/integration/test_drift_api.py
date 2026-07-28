"""
Integration tests for GET /drift/{model_version}.

The Mann-Whitney statistics themselves are covered by tests/test_drift.py
against the pure function. Here we only check that the route wires query
params and the DB-backed path through correctly.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_drift_reports_insufficient_data_for_unseen_model(client):
    response = client.get("/drift/model-with-no-history")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "model-with-no-history"
    assert body["status"] == "insufficient_data"
    assert body["drift_detected"] is False
    assert body["p_value"] is None
    assert body["current_count"] == 0
    assert body["baseline_count"] == 0


def test_drift_accepts_custom_window_query_params(client):
    response = client.get(
        "/drift/model-with-no-history",
        params={"current_hours": 48, "baseline_days": 14},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "params",
    [
        {"current_hours": 0},
        {"current_hours": 200},
        {"baseline_days": 0},
        {"baseline_days": 91},
    ],
)
def test_drift_rejects_out_of_range_window_params(client, params):
    response = client.get("/drift/model-with-no-history", params=params)
    assert response.status_code == 422
