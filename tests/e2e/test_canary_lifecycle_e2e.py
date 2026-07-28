"""
End-to-end test for the Phase 4 provider-change canary.

There's no HTTP endpoint for triggering a canary run (it's invoked by
scripts/run_canary.py on a nightly schedule — see app/core/canary.py), so
this test drives run_canary() directly against the same database session
backing the HTTP client, then verifies the result is visible where a real
operator would look for it: the /metrics Prometheus endpoint.
"""

from __future__ import annotations

import pytest

from app.core.canary import run_canary
from app.db.models.centroid_history import CentroidHistory

pytestmark = pytest.mark.e2e


def _create_case(client, prompt: str) -> str:
    response = client.post("/cases", json={"prompt": prompt})
    assert response.status_code == 201
    return response.json()["case_id"]


def test_canary_run_persists_and_surfaces_in_metrics(client, db_session):
    # run_canary() with no golden_case_ids queries every golden case in the
    # database — including the shared dev DB's real seed data. Scope it to
    # exactly the two cases this test creates so response_count is deterministic.
    case_ids = [
        _create_case(client, "What are the symptoms of the common cold?"),
        _create_case(client, "How do I safely store leftover cooked rice?"),
    ]

    # First run: no prior centroid to compare against, so drift is 0.0 by
    # definition and no alert fires.
    first = run_canary(db_session, provider="mock", golden_case_ids=case_ids, alert_threshold=0.05)
    assert first.provider == "mock"
    assert first.drift_score == 0.0
    assert first.alert_sent is False
    assert first.response_count == 2

    # Second run: the mock provider always returns identical text, so the
    # centroid doesn't move and drift stays at 0.0 — a deterministic canary
    # "all clear" outcome.
    second = run_canary(db_session, provider="mock", golden_case_ids=case_ids, alert_threshold=0.05)
    assert second.drift_score == 0.0
    assert second.alert_sent is False

    history_rows = db_session.query(CentroidHistory).filter_by(provider="mock").all()
    assert len(history_rows) == 2

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'driftscope_canary_centroid_drift{provider="mock"} 0.000000' in metrics.text
