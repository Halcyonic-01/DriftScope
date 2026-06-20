"""
app/core/eval_service.py

Core evaluation logic — runs a golden case against an LLM and scores the result.

FLOW:
  1. Fetch the golden case from the DB
  2. Call the chosen LLM with the prompt
  3. Embed the LLM's response text
  4. Build a semantic reference string from the case contract and embed that too
  5. cosine_score = cosine_sim(response_embedding, reference_embedding)
  6. composite_score = cosine_score  (judge_score added in Phase 2)
  7. Persist the EvalResult to the DB and return it

WHY build a semantic reference instead of joining keywords?
  We don't store a "perfect golden answer" — that's too rigid.
  Instead we store what TOPICS the answer should cover and what rules it must
  obey. Embedding a full natural-language contract gives the embedding model
  enough context to compare against a real response.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.core.embeddings import embed, cosine_sim
from app.core.judge import (
    composite_score as calculate_composite_score,
    judge_response_against_rules,
    should_invoke_judge,
)
from app.core.llm import get_client
from app.db.models.eval_result import EvalResult
from app.db.models.golden_case import GoldenCase

logger = logging.getLogger(__name__)


def run_eval(
    db: Session,
    case: GoldenCase,
    model_version: str,
    provider: str = "mock",
) -> EvalResult:
    """
    Execute a full evaluation of one golden case against an LLM.

    Args:
        db:            Active SQLAlchemy session.
        case:          The GoldenCase to evaluate.
        model_version: Label for the LLM version being tested.
        provider:      LLM provider ("gemini", "ollama", "mock").

    Returns:
        A persisted EvalResult row with all scores filled in.
    """
    logger.info(
        "Running eval for case=%s, provider=%s, version=%s",
        case.case_id, provider, model_version,
    )

    # ── Step 1: Call the LLM with the prompt ──────────────────────────────────
    llm = get_client(provider)
    llm_response = llm.complete(case.prompt)
    response_text = llm_response.text

    logger.debug("LLM response (first 120 chars): %s", response_text[:120])

    # ── Step 2: Compute cosine score ──────────────────────────────────────────
    cosine_score: Optional[float] = None

    reference_text = build_reference_text(
        prompt=case.prompt,
        expected_topics=case.expected_topics,
        safety_rules=case.safety_rules,
        domain=case.domain,
    )

    if reference_text:
        response_vec = embed(response_text)
        reference_vec = embed(reference_text)

        cosine_score = cosine_sim(response_vec, reference_vec)
        logger.info("cosine_score=%.4f", cosine_score)
    else:
        logger.warning(
            "Case %s has no expected_topics — cosine_score will be None", case.case_id
        )

    # ── Step 3: Optional LLM-as-judge ────────────────────────────────────────
    judge_score: Optional[float] = None
    judge_reason: Optional[str] = None

    if should_invoke_judge(cosine_score, case.safety_rules, case.domain):
        logger.info("Invoking LLM judge for case=%s", case.case_id)
        judge_score, judge_reason = judge_response_against_rules(
            client=llm,
            response=response_text,
            expected_topics=case.expected_topics,
            safety_rules=case.safety_rules,
        )
        logger.info("judge_score=%s", f"{judge_score:.4f}" if judge_score is not None else "None")
    else:
        logger.info("Skipping judge for case=%s due to cost guard", case.case_id)

    # ── Step 4: Composite score ───────────────────────────────────────────────
    composite_score = calculate_composite_score(cosine_score, judge_score)

    # ── Step 5: Persist result ────────────────────────────────────────────────
    result = EvalResult(
        result_id=uuid.uuid4(),
        case_id=case.case_id,
        model_version=model_version,
        response_text=response_text,
        cosine_score=cosine_score,
        judge_score=judge_score,
        composite_score=composite_score,
        judge_reason=judge_reason,
        provider=provider,
    )

    db.add(result)
    db.flush()   # write to DB but don't commit yet (route commits)

    logger.info(
        "EvalResult persisted: result_id=%s, composite=%.4f",
        result.result_id, composite_score or 0.0,
    )

    return result


def build_reference_text(
    prompt: str,
    expected_topics: list[str] | None = None,
    safety_rules: list[str] | None = None,
    domain: str | None = None,
) -> str | None:
    """
    Build the semantic target used for cosine scoring.

    The reference is intentionally a natural-language behavioral contract, not
    a keyword bag. Sentence embeddings compare meaning better when both sides
    look like coherent text.
    """
    topics = _clean_list(expected_topics)
    rules = _clean_list(safety_rules)

    if not topics and not rules:
        return None

    lines = [
        f"User prompt: {prompt.strip()}",
    ]
    if domain:
        lines.append(f"Domain: {domain.strip()}")

    if topics:
        lines.append(
            "A good response should meaningfully cover: "
            f"{_format_items(topics)}."
        )

    if rules:
        lines.append(
            "The response must follow these safety and quality rules: "
            f"{_format_items(rules)}."
        )

    return "\n".join(lines)


def _clean_list(values: list[str] | None) -> list[str]:
    return [value.strip() for value in values or [] if value and value.strip()]


def _format_items(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"
