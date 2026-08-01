"""
tests/test_judge_batching.py

The judge used to make one provider call per rule, which was the biggest
consumer of free-tier quota: a case with topic coverage plus two safety
rules cost three judge calls on top of the generation call. Batching asks
about every rule in one call.

The quality contract these tests defend: batching must change only the
number of calls, never what gets evaluated. Every rule still gets its own
pass/fail and its own reason.
"""

from __future__ import annotations

import json

import pytest

from app.core.judge import (
    JudgeDecision,
    _parse_batched_judge_json,
    judge_response_against_rules,
)
from app.core.llm.base import LLMResponse


class RecordingClient:
    """Returns a canned judge payload and counts calls."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def complete(self, prompt: str, response_mime_type: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(text=self.payload, provider="fake", model="fake-model")


def _payload(*entries):
    return json.dumps({"results": [
        {"id": i, "pass": p, "reason": r} for i, p, r in entries
    ]})


def test_all_rules_judged_in_a_single_call():
    """Three rules, one call — this is the whole point of the change."""
    client = RecordingClient(_payload(
        ("topics", True, "Covers the topics."),
        ("safety_1", True, "Advises a doctor."),
        ("safety_2", False, "Suggests an unsafe dose."),
    ))

    score, reason = judge_response_against_rules(
        client,
        "some response",
        expected_topics=["dosage"],
        safety_rules=["Must advise consulting a doctor", "Must not exceed 400mg"],
    )

    assert len(client.prompts) == 1                      # was 3 before batching
    assert score == pytest.approx(2 / 3)                 # every rule still counted
    assert "topics: pass - Covers the topics." in reason
    assert "safety_1: pass - Advises a doctor." in reason
    assert "safety_2: fail - Suggests an unsafe dose." in reason


def test_every_rule_appears_in_the_prompt():
    """Batching must not quietly drop a rule from what the model sees."""
    client = RecordingClient(_payload(
        ("topics", True, "ok"), ("safety_1", True, "ok"), ("safety_2", True, "ok"),
    ))

    judge_response_against_rules(
        client, "resp",
        expected_topics=["hydration"],
        safety_rules=["Must advise a doctor", "Must not exceed 400mg"],
    )

    prompt = client.prompts[0]
    assert 'id: "topics"' in prompt
    assert 'id: "safety_1"' in prompt
    assert 'id: "safety_2"' in prompt
    assert "Must advise a doctor" in prompt
    assert "Must not exceed 400mg" in prompt


def test_score_matches_per_rule_semantics():
    """A half-failing batch scores 0.5, exactly as separate calls would."""
    client = RecordingClient(_payload(
        ("topics", True, "ok"), ("safety_1", False, "violates"),
    ))

    score, _ = judge_response_against_rules(
        client, "resp", expected_topics=["x"], safety_rules=["y"],
    )

    assert score == pytest.approx(0.5)


def test_no_call_made_when_there_are_no_rules():
    client = RecordingClient(_payload())

    score, reason = judge_response_against_rules(client, "resp", [], [])

    assert (score, reason) == (None, None)
    assert client.prompts == []


# ── Parser guards ────────────────────────────────────────────────────────

def test_missing_verdict_is_rejected():
    """
    A dropped rule would shrink the pass-rate denominator and silently
    inflate the score — the one failure mode a quality monitor must not
    have. It must raise instead.
    """
    with pytest.raises(ValueError, match="omitted verdicts for: safety_1"):
        _parse_batched_judge_json(
            _payload(("topics", True, "ok")),
            expected_ids=["topics", "safety_1"],
        )


def test_unexpected_ids_are_ignored():
    decisions = _parse_batched_judge_json(
        _payload(("topics", True, "ok"), ("hallucinated", False, "nope")),
        expected_ids=["topics"],
    )
    assert set(decisions) == {"topics"}


def test_invalid_json_is_rejected():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_batched_judge_json("not json", expected_ids=["topics"])


def test_missing_results_list_is_rejected():
    with pytest.raises(ValueError, match='"results" list'):
        _parse_batched_judge_json('{"pass": true}', expected_ids=["topics"])


def test_non_boolean_pass_is_rejected():
    payload = json.dumps({"results": [{"id": "topics", "pass": "yes", "reason": "ok"}]})
    with pytest.raises(ValueError, match='needs a boolean "pass"'):
        _parse_batched_judge_json(payload, expected_ids=["topics"])


def test_empty_reason_is_rejected():
    payload = json.dumps({"results": [{"id": "topics", "pass": True, "reason": "  "}]})
    with pytest.raises(ValueError, match='needs a non-empty "reason"'):
        _parse_batched_judge_json(payload, expected_ids=["topics"])


def test_parses_a_well_formed_batch():
    decisions = _parse_batched_judge_json(
        _payload(("topics", True, "Good."), ("safety_1", False, "Bad.")),
        expected_ids=["topics", "safety_1"],
    )
    assert decisions["topics"] == JudgeDecision(passed=True, reason="Good.")
    assert decisions["safety_1"].score == 0.0
