"""
Tests for Phase 4 provider-change canary logic.
"""

from __future__ import annotations

import uuid

import numpy as np

from app.core import canary
from app.core.canary import centroid_drift, compute_centroid, run_canary
from app.core.llm.base import LLMResponse
from app.db.models.golden_case import GoldenCase


class FakeEmbeddingModel:
    def encode(self, responses, normalize_embeddings=True):
        values = []
        for response in responses:
            if response == "east":
                values.append([1.0, 0.0])
            elif response == "north":
                values.append([0.0, 1.0])
            else:
                values.append([0.6, 0.8])
        return np.asarray(values)


class FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def complete(self, prompt, response_mime_type=None):
        return LLMResponse(
            text=self.response_text,
            provider="mock",
            model="mock-model",
            tokens_used=1,
        )


class FakeQuery:
    def __init__(self, values):
        self.values = values

    def filter(self, *args):
        return self

    def order_by(self, *args):
        return self

    def limit(self, value):
        self.values = self.values[:value]
        return self

    def all(self):
        return self.values


class FakeSession:
    def __init__(self, cases, previous_centroids=None):
        self.cases = cases
        self.previous_centroids = previous_centroids or []
        self.added = []

    def query(self, model):
        if model is GoldenCase:
            return FakeQuery(self.cases)
        return FakeQuery([(centroid,) for centroid in self.previous_centroids])

    def add(self, value):
        self.added.append(value)

    def flush(self):
        for value in self.added:
            if value.id is None:
                value.id = uuid.uuid4()


def test_centroid_drift_is_zero_for_identical_vectors():
    assert centroid_drift([1.0, 0.0], [1.0, 0.0]) == 0.0


def test_centroid_drift_is_one_for_orthogonal_vectors():
    assert centroid_drift([1.0, 0.0], [0.0, 1.0]) == 1.0


def test_compute_centroid_averages_response_embeddings(monkeypatch):
    monkeypatch.setattr(canary, "get_model", lambda: FakeEmbeddingModel())

    centroid = compute_centroid(["east", "north"])

    assert np.allclose(centroid, [0.5, 0.5])


def test_run_canary_stores_snapshot_and_sends_alert(monkeypatch):
    case = GoldenCase(prompt="Prompt", expected_topics=[], safety_rules=[])
    db = FakeSession(cases=[case], previous_centroids=[[0.0, 1.0]])

    monkeypatch.setattr(canary, "get_client", lambda provider: FakeLLM("east"))
    monkeypatch.setattr(canary, "get_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(canary, "send_configured_email_alert", lambda *args: True)

    result = run_canary(db, provider="mock", alert_threshold=0.05)

    assert result.provider == "mock"
    assert result.drift_score == 1.0
    assert result.alert_sent is True
    assert result.response_count == 1
    assert db.added[0].provider == "mock"
    assert db.added[0].centroid == [1.0, 0.0]


def test_run_canary_honors_case_limit(monkeypatch):
    cases = [
        GoldenCase(prompt="Prompt 1", expected_topics=[], safety_rules=[]),
        GoldenCase(prompt="Prompt 2", expected_topics=[], safety_rules=[]),
    ]
    db = FakeSession(cases=cases)

    monkeypatch.setattr(canary, "get_client", lambda provider: FakeLLM("east"))
    monkeypatch.setattr(canary, "get_model", lambda: FakeEmbeddingModel())

    result = run_canary(db, provider="mock", case_limit=1)

    assert result.response_count == 1


def test_run_canary_paces_calls_between_cases(monkeypatch):
    """
    The canary makes one call per case with no natural gap, so an unpaced
    run fires every request at once and trips a per-minute provider limit.
    """
    cases = [
        GoldenCase(prompt=f"Prompt {i}", expected_topics=[], safety_rules=[])
        for i in range(3)
    ]
    db = FakeSession(cases=cases)
    sleeps = []

    monkeypatch.setattr(canary, "get_client", lambda provider: FakeLLM("east"))
    monkeypatch.setattr(canary, "get_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(canary.time, "sleep", lambda s: sleeps.append(s))

    canary.run_canary(db, provider="mock", delay_seconds=5.0)

    # One gap between each pair of cases, never before the first.
    assert sleeps == [5.0, 5.0]


def test_run_canary_delay_can_be_disabled(monkeypatch):
    cases = [
        GoldenCase(prompt=f"Prompt {i}", expected_topics=[], safety_rules=[])
        for i in range(3)
    ]
    db = FakeSession(cases=cases)
    sleeps = []

    monkeypatch.setattr(canary, "get_client", lambda provider: FakeLLM("east"))
    monkeypatch.setattr(canary, "get_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(canary.time, "sleep", lambda s: sleeps.append(s))

    canary.run_canary(db, provider="mock", delay_seconds=0)

    assert sleeps == []
