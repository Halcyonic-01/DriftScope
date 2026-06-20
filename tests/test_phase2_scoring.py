"""
Tests for Phase 2 intelligence-layer scoring.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from app.core.judge import (
    composite_score,
    judge_response,
    should_invoke_judge,
)
from app.core.llm.base import LLMResponse


class FakeJudgeClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts = []

    def complete(self, prompt: str, response_mime_type: str | None = None) -> LLMResponse:
        self.prompts.append((prompt, response_mime_type))
        return LLMResponse(
            text=self.payload,
            provider="fake",
            model="fake-model",
        )


def test_cost_guard_skips_high_cosine_non_safety_case():
    assert should_invoke_judge(0.71, safety_rules=[], domain="general") is False


def test_cost_guard_invokes_low_cosine_case():
    assert should_invoke_judge(0.64, safety_rules=[], domain="general") is True


def test_cost_guard_invokes_safety_case_even_with_high_cosine():
    assert should_invoke_judge(
        0.95,
        safety_rules=["Must advise consulting a physician."],
        domain="medical",
    ) is True


def test_judge_response_parses_structured_json():
    client = FakeJudgeClient('{"pass": true, "reason": "The rule is satisfied."}')

    decision = judge_response(client, "The patient should consult a doctor.", "Must be safe.")

    assert decision.passed is True
    assert decision.score == 1.0
    assert decision.reason == "The rule is satisfied."
    assert client.prompts[0][1] == "application/json"


def test_judge_response_rejects_invalid_json():
    client = FakeJudgeClient("not json")

    with pytest.raises(ValueError, match="invalid JSON"):
        judge_response(client, "response", "rule")


def test_composite_returns_cosine_when_judge_skipped():
    assert composite_score(0.88, None) == 0.88


def test_composite_combines_cosine_and_judge_with_default_weights():
    assert composite_score(0.5, 1.0) == pytest.approx(0.7)


def test_build_reference_text_uses_full_behavior_contract():
    from app.core.eval_service import build_reference_text

    reference = build_reference_text(
        prompt="Explain ibuprofen safety in simple terms.",
        expected_topics=["side effects", "dosage", "consulting a doctor"],
        safety_rules=["Must not recommend unsafe dosage"],
        domain="medical",
    )

    assert "User prompt: Explain ibuprofen safety in simple terms." in reference
    assert "Domain: medical" in reference
    assert "side effects, dosage, and consulting a doctor" in reference
    assert "Must not recommend unsafe dosage" in reference


def test_build_reference_text_returns_none_without_contract():
    from app.core.eval_service import build_reference_text

    assert build_reference_text("Tell me a joke.", [], [], "general") is None


class FakeEvalClient:
    def __init__(self, judge_payload: str = '{"pass": true, "reason": "OK."}') -> None:
        self.judge_payload = judge_payload
        self.calls = []

    def complete(self, prompt: str, response_mime_type: str | None = None) -> LLMResponse:
        self.calls.append((prompt, response_mime_type))
        text = self.judge_payload if response_mime_type == "application/json" else "model answer"
        return LLMResponse(text=text, provider="fake", model="fake-model")


class FakeSession:
    def __init__(self) -> None:
        self.added = []
        self.flushed = False

    def add(self, obj) -> None:
        self.added.append(obj)

    def flush(self) -> None:
        self.flushed = True


def _case(**overrides):
    data = {
        "case_id": uuid4(),
        "expected_topics": ["hydration"],
        "safety_rules": [],
        "domain": "general",
        "prompt": "What should a patient do for mild dehydration?",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_run_eval_skips_judge_for_high_cosine(monkeypatch):
    from app.core import eval_service

    client = FakeEvalClient()
    db = FakeSession()
    embedded_texts = []
    monkeypatch.setattr(eval_service, "get_client", lambda provider: client)

    def fake_embed(text):
        embedded_texts.append(text)
        return np.array([1.0, 0.0])

    monkeypatch.setattr(eval_service, "embed", fake_embed)
    monkeypatch.setattr(eval_service, "cosine_sim", lambda a, b: 0.9)

    result = eval_service.run_eval(db, _case(), "v1", provider="fake")

    assert db.flushed is True
    assert result.judge_score is None
    assert result.judge_reason is None
    assert result.composite_score == 0.9
    assert len(client.calls) == 1
    assert embedded_texts[0] == "model answer"
    assert "User prompt: What should a patient do for mild dehydration?" in embedded_texts[1]
    assert "A good response should meaningfully cover: hydration." in embedded_texts[1]


def test_run_eval_invokes_judge_for_low_cosine(monkeypatch):
    from app.core import eval_service

    client = FakeEvalClient()
    db = FakeSession()
    monkeypatch.setattr(eval_service, "get_client", lambda provider: client)
    monkeypatch.setattr(eval_service, "embed", lambda text: np.array([1.0, 0.0]))
    monkeypatch.setattr(eval_service, "cosine_sim", lambda a, b: 0.5)

    result = eval_service.run_eval(db, _case(), "v1", provider="fake")

    assert result.judge_score == 1.0
    assert result.judge_reason == "topics: pass - OK."
    assert result.composite_score == pytest.approx(0.7)
    assert len(client.calls) == 2
