"""
Integration test for scripts/run_eval.py — the CI quality-gate script.

scripts.run_eval.run_suite() uses app.db.session.get_db() internally
(a fresh, self-committing session), which would bypass the SAVEPOINT
rollback sandbox and write real rows to whatever database it's pointed
at. We monkeypatch it to route through the test's own transactional
db_session instead, so this test is safe to run against a shared DB
while still exercising the real code path end-to-end.
"""

from __future__ import annotations

import contextlib
import json

import pytest

from app.db.models.eval_result import EvalResult
from app.db.models.golden_case import GoldenCase
from scripts import run_eval
from scripts.seed import SEED_CASES

pytestmark = pytest.mark.integration


@pytest.fixture()
def patched_get_db(monkeypatch, db_session):
    @contextlib.contextmanager
    def _fake_get_db():
        yield db_session
        db_session.commit()

    monkeypatch.setattr(run_eval, "get_db", _fake_get_db)
    return db_session


def test_run_suite_scores_every_seed_case_with_a_contract(patched_get_db):
    scores = run_eval.run_suite(model_version="test-ci-baseline", provider="mock")

    expected_scored_cases = [c for c in SEED_CASES if c.get("expected_topics") or c.get("safety_rules")]
    assert len(scores) == len(expected_scored_cases) == len(SEED_CASES)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_run_suite_is_deterministic_across_runs(patched_get_db):
    first = run_eval.run_suite(model_version="test-ci-baseline-a", provider="mock")
    second = run_eval.run_suite(model_version="test-ci-baseline-b", provider="mock")

    # The embedding forward pass isn't guaranteed bit-identical even within
    # the same process (BLAS thread scheduling etc. can introduce ~1e-6
    # noise), so compare with a tolerance rather than exact float equality.
    assert first == pytest.approx(second, abs=1e-3)


def test_run_suite_reuses_existing_golden_cases_instead_of_duplicating(patched_get_db):
    run_eval.run_suite(model_version="test-ci-baseline-1", provider="mock")
    run_eval.run_suite(model_version="test-ci-baseline-2", provider="mock")

    seed_prompts = [c["prompt"] for c in SEED_CASES]
    case_count = (
        patched_get_db.query(GoldenCase)
        .filter(GoldenCase.prompt.in_(seed_prompts))
        .count()
    )
    assert case_count == len(SEED_CASES)

    result_count = (
        patched_get_db.query(EvalResult)
        .filter(EvalResult.model_version.in_(["test-ci-baseline-1", "test-ci-baseline-2"]))
        .count()
    )
    assert result_count == 2 * len(SEED_CASES)


def test_committed_baseline_matches_current_suite_output(patched_get_db):
    """
    Guards against the checked-in baseline_scores.json silently drifting
    out of sync with what the suite actually produces today.

    Uses a tolerance rather than exact equality: the embedding forward
    pass can differ by ~1e-6 across platforms/BLAS backends (e.g. macOS
    MPS vs. Linux CPU in CI) without indicating an actual regression.
    A real scoring/logic regression would shift scores by orders of
    magnitude more than this tolerance.
    """
    current_scores = run_eval.run_suite(model_version="test-ci-baseline", provider="mock")
    baseline_scores = json.loads(run_eval.BASELINE_PATH.read_text(encoding="utf-8"))

    assert current_scores == pytest.approx(baseline_scores, abs=1e-3)
