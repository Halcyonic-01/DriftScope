"""
Integration tests for scripts/run_monitor.py — the scheduled eval job
that keeps eval_results inside the 24h metrics window.

As in test_run_eval_script.py, get_db() is patched onto the test's
transactional session so nothing is written to a real database.
"""

from __future__ import annotations

import contextlib

import pytest

from app.db.models.eval_result import EvalResult
from app.db.models.golden_case import GoldenCase
from scripts import run_monitor

pytestmark = pytest.mark.integration


@pytest.fixture()
def patched_get_db(monkeypatch, db_session):
    @contextlib.contextmanager
    def _fake_get_db():
        yield db_session

    # run_monitor commits per case; the SAVEPOINT fixture turns those into
    # savepoint restarts, so they still roll back at teardown.
    monkeypatch.setattr(run_monitor, "get_db", _fake_get_db)
    return db_session


@pytest.fixture()
def one_case(db_session):
    case = GoldenCase(
        prompt="Explain the side effects of ibuprofen in simple terms.",
        expected_topics=["side effects", "dosage"],
        safety_rules=[],
        domain="monitor-test-domain",
    )
    db_session.add(case)
    db_session.flush()
    return case


def _run(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["run_monitor.py", *argv])
    run_monitor.main()


def test_monitor_records_results_for_each_case(patched_get_db, one_case, monkeypatch, capsys):
    _run(monkeypatch, [
        "--provider", "mock",
        "--model-version", "monitor-test-v1",
        "--domain", "monitor-test-domain",
    ])

    rows = (
        patched_get_db.query(EvalResult)
        .filter(EvalResult.model_version == "monitor-test-v1")
        .all()
    )
    assert len(rows) == 1
    assert rows[0].composite_score is not None

    out = capsys.readouterr().out
    assert '"cases_succeeded": 1' in out
    assert '"cases_failed": 0' in out


def test_monitor_model_version_defaults_to_provider(patched_get_db, one_case, monkeypatch):
    _run(monkeypatch, ["--provider", "mock", "--domain", "monitor-test-domain"])

    assert (
        patched_get_db.query(EvalResult).filter(EvalResult.model_version == "mock").count() == 1
    )


def test_monitor_isolates_per_case_failures(patched_get_db, one_case, monkeypatch, capsys):
    """One provider error must not abort the whole run and leave a hole."""
    from app.core import eval_service

    def boom(*args, **kwargs):
        raise RuntimeError("provider quota exhausted")

    monkeypatch.setattr(run_monitor, "run_eval", boom)

    with pytest.raises(SystemExit):  # 1/1 failed, past the 50% threshold
        _run(monkeypatch, [
            "--provider", "mock", "--domain", "monitor-test-domain",
        ])

    out = capsys.readouterr().out
    assert '"cases_failed": 1' in out


def test_monitor_exits_when_no_cases_match(patched_get_db, monkeypatch):
    with pytest.raises(SystemExit, match="No golden cases found"):
        _run(monkeypatch, ["--provider", "mock", "--domain", "no-such-domain-xyz"])
