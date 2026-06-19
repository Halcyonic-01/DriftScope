"""
app/core/eval_service.py

Core evaluation logic — runs a golden case against an LLM and scores the result.

FLOW:
  1. Fetch the golden case from the DB
  2. Call the chosen LLM with the prompt
  3. Embed the LLM's response text
  4. Build a "reference" string from expected_topics and embed that too
  5. cosine_score = cosine_sim(response_embedding, reference_embedding)
  6. composite_score = cosine_score  (judge_score added in Phase 2)
  7. Persist the EvalResult to the DB and return it

WHY embed expected_topics as the reference?
  We don't store a "perfect golden answer" — that's too rigid.
  Instead we store what TOPICS the answer should cover.
  Joining topics into a sentence ("side effects dosage contraindications")
  and embedding that gives us a semantic target vector.
  If the LLM response covers those topics, its embedding will be geometrically
  close to the reference — high cosine score.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.core.embeddings import embed, cosine_sim
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

    if case.expected_topics:
        # Build a reference string from the expected topics
        reference_text = " ".join(case.expected_topics)

        response_vec = embed(response_text)
        reference_vec = embed(reference_text)

        cosine_score = cosine_sim(response_vec, reference_vec)
        logger.info("cosine_score=%.4f", cosine_score)
    else:
        logger.warning(
            "Case %s has no expected_topics — cosine_score will be None", case.case_id
        )

    # ── Step 3: Composite score ───────────────────────────────────────────────
    # Phase 1: composite = cosine only (judge added in Phase 2)
    composite_score = cosine_score

    # ── Step 4: Persist result ────────────────────────────────────────────────
    result = EvalResult(
        result_id=uuid.uuid4(),
        case_id=case.case_id,
        model_version=model_version,
        response_text=response_text,
        cosine_score=cosine_score,
        judge_score=None,        # Phase 2
        composite_score=composite_score,
        judge_reason=None,       # Phase 2
        provider=provider,
    )

    db.add(result)
    db.flush()   # write to DB but don't commit yet (route commits)

    logger.info(
        "EvalResult persisted: result_id=%s, composite=%.4f",
        result.result_id, composite_score or 0.0,
    )

    return result
