"""
Integration tests for canary baseline isolation.

A centroid is the mean embedding of the prompts it was built from, so
snapshots built from different prompt sets are not comparable. Mixing a
5-case snapshot into the baseline for a 20-case run produced drift scores
several times the 0.05 alert threshold that reflected the change of
inputs, not any change in the model.

These use the real database because the behaviour under test is a SQL
filter, which the fake session in tests/test_canary.py cannot exercise.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.canary import get_rolling_centroid
from app.db.models.centroid_history import CentroidHistory

pytestmark = pytest.mark.integration


def _snapshot(db, provider, centroid, case_count):
    db.add(
        CentroidHistory(
            provider=provider,
            centroid=centroid,
            drift_score=0.0,
            case_count=case_count,
        )
    )
    db.flush()


def test_baseline_excludes_snapshots_from_a_different_case_count(db_session):
    provider = "isolation-test"
    # Two 20-case snapshots pointing one way...
    _snapshot(db_session, provider, [1.0, 0.0], case_count=20)
    _snapshot(db_session, provider, [1.0, 0.0], case_count=20)
    # ...and a 5-case snapshot pointing the opposite way.
    _snapshot(db_session, provider, [0.0, 1.0], case_count=5)

    baseline = get_rolling_centroid(db_session, provider, case_count=20)

    # The 5-case snapshot must not drag the 20-case baseline sideways.
    assert np.allclose(baseline, [1.0, 0.0])


def test_baseline_without_case_count_still_averages_everything(db_session):
    """The unfiltered path is unchanged for callers that don't specify."""
    provider = "isolation-test-unfiltered"
    _snapshot(db_session, provider, [1.0, 0.0], case_count=20)
    _snapshot(db_session, provider, [0.0, 1.0], case_count=5)

    baseline = get_rolling_centroid(db_session, provider)

    assert np.allclose(baseline, [0.5, 0.5])


def test_no_baseline_when_only_other_sizes_exist(db_session):
    """
    A first run at a new case count has nothing valid to compare against,
    so it must report no baseline rather than borrow a mismatched one.
    """
    provider = "isolation-test-newsize"
    _snapshot(db_session, provider, [1.0, 0.0], case_count=5)

    assert get_rolling_centroid(db_session, provider, case_count=20) is None


def test_legacy_rows_with_null_case_count_are_excluded(db_session):
    """
    Rows recorded before case_count existed have an unknown prompt set,
    so they must not be assumed to match.
    """
    provider = "isolation-test-legacy"
    _snapshot(db_session, provider, [0.0, 1.0], case_count=None)
    _snapshot(db_session, provider, [1.0, 0.0], case_count=20)

    baseline = get_rolling_centroid(db_session, provider, case_count=20)

    assert np.allclose(baseline, [1.0, 0.0])
