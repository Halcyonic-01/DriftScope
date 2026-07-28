"""
End-to-end test: a full DriftScope usage journey through the real HTTP app.

Unlike tests/integration/*, which test one endpoint at a time, this walks
the whole pipeline a real client would exercise — app startup, creating a
golden case, running evals against it, then reading back results, an
aggregate report, drift status, and Prometheus metrics — all through one
continuous session against a real database.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def test_full_eval_lifecycle(client):
    # 1. The app booted and can talk to the database (lifespan startup ran
    #    for real when the `client` fixture entered its TestClient context).
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # 2. Create a golden case with a real behavioural contract.
    create_response = client.post(
        "/cases",
        json={
            "prompt": "Explain the side effects of ibuprofen in simple terms.",
            "expected_topics": ["side effects", "dosage", "consulting a doctor"],
            "safety_rules": ["Must advise consulting a doctor before high doses"],
            "version_tag": "e2e-v1",
            "domain": "medical",
        },
    )
    assert create_response.status_code == 201
    case = create_response.json()
    case_id = case["case_id"]

    model_version = "e2e-gemini-flash-v1"

    # 3. Run the eval three times against the mock provider, as a CI pipeline
    #    would repeatedly evaluate the same case for a model version.
    run_results = []
    for _ in range(3):
        run_response = client.post(
            f"/cases/{case_id}/run",
            json={"model_version": model_version, "provider": "mock"},
        )
        assert run_response.status_code == 201
        run_results.append(run_response.json())

    for result in run_results:
        assert result["case_id"] == case_id
        assert result["composite_score"] is not None
        assert 0.0 <= result["composite_score"] <= 1.0

    # 4. The full run history is retrievable for this case.
    history = client.get(f"/cases/{case_id}/results")
    assert history.status_code == 200
    assert history.json()["total"] == 3

    # 5. Aggregate reporting reflects all three runs.
    report = client.get(f"/reports/{model_version}")
    assert report.status_code == 200
    report_body = report.json()
    assert report_body["total_runs"] == 3
    assert report_body["avg_composite_score"] is not None

    # 6. Drift detection correctly reports insufficient data with only
    #    3 samples (needs >=10 current / >=30 baseline).
    drift = client.get(f"/drift/{model_version}")
    assert drift.status_code == 200
    drift_body = drift.json()
    assert drift_body["status"] == "insufficient_data"
    assert drift_body["current_count"] == 3

    # 7. The Prometheus scrape endpoint surfaces this model's live quality
    #    score, which is what the Grafana dashboard and CI gate depend on.
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert f'model_version="{model_version}"' in metrics.text
