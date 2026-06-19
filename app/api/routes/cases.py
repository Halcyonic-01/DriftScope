"""
app/api/routes/cases.py

Route handlers for the /cases endpoints.

Routes are kept THIN — they only handle:
  1. Receiving the HTTP request
  2. Validating input (Pydantic does this automatically)
  3. Calling the service layer
  4. Returning the HTTP response

All business logic lives in app/core/eval_service.py
All DB queries live here as simple SQLAlchemy calls.
"""

from __future__ import annotations

import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.schemas import (
    CreateCaseRequest,
    CaseResponse,
    CaseListResponse,
    EvalResultResponse,
    EvalResultListResponse,
    RunEvalRequest,
)
from app.core.eval_service import run_eval
from app.db.models.golden_case import GoldenCase
from app.db.models.eval_result import EvalResult
from app.db.session import get_db_session

logger = logging.getLogger(__name__)

# All routes in this file will be prefixed with /cases
router = APIRouter(prefix="/cases", tags=["Cases"])


# ── POST /cases ───────────────────────────────────────────────────────────────

@router.post("", response_model=CaseResponse, status_code=201)
def create_case(
    body: CreateCaseRequest,
    db: Session = Depends(get_db_session),
):
    """
    Create a new golden test case.

    A golden case stores the behavioural contract for a prompt:
    what topics the response should cover, and what rules it must follow.
    """
    case = GoldenCase(
        case_id=uuid.uuid4(),
        prompt=body.prompt,
        expected_topics=body.expected_topics or [],
        safety_rules=body.safety_rules or [],
        version_tag=body.version_tag,
        domain=body.domain,
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    logger.info("Created golden case: %s", case.case_id)
    return case


# ── GET /cases ────────────────────────────────────────────────────────────────

@router.get("", response_model=CaseListResponse)
def list_cases(
    page: int = Query(default=1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    domain: Optional[str] = Query(default=None, description="Filter by domain"),
    db: Session = Depends(get_db_session),
):
    """
    List all golden test cases with optional domain filter and pagination.
    """
    query = db.query(GoldenCase)

    if domain:
        query = query.filter(GoldenCase.domain == domain)

    total = query.count()
    cases = (
        query
        .order_by(GoldenCase.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return CaseListResponse(
        items=cases,
        total=total,
        page=page,
        page_size=page_size,
    )


# ── GET /cases/{case_id} ──────────────────────────────────────────────────────

@router.get("/{case_id}", response_model=CaseResponse)
def get_case(
    case_id: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    """
    Retrieve a single golden test case by its UUID.
    Returns 404 if not found.
    """
    case = db.query(GoldenCase).filter(GoldenCase.case_id == case_id).first()

    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    return case


# ── POST /cases/{case_id}/run ─────────────────────────────────────────────────

@router.post("/{case_id}/run", response_model=EvalResultResponse, status_code=201)
def run_case(
    case_id: uuid.UUID,
    body: RunEvalRequest,
    db: Session = Depends(get_db_session),
):
    """
    Run an evaluation on a golden case against an LLM.

    Flow:
      1. Fetch the golden case
      2. Call the specified LLM with the prompt
      3. Embed the response and compute cosine similarity vs expected topics
      4. Persist and return the EvalResult
    """
    case = db.query(GoldenCase).filter(GoldenCase.case_id == case_id).first()

    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    try:
        result = run_eval(
            db=db,
            case=case,
            model_version=body.model_version,
            provider=body.provider,
        )
        db.commit()
        db.refresh(result)
        return result

    except ValueError as exc:
        # e.g. unknown provider name from factory
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Eval failed for case=%s: %s", case_id, exc)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Eval failed: {exc}")


# ── GET /cases/{case_id}/results ──────────────────────────────────────────────

@router.get("/{case_id}/results", response_model=EvalResultListResponse)
def get_case_results(
    case_id: uuid.UUID,
    db: Session = Depends(get_db_session),
):
    """
    Get the full eval history for a golden case.
    Returns all runs ordered by most recent first.
    """
    case = db.query(GoldenCase).filter(GoldenCase.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    results = (
        db.query(EvalResult)
        .filter(EvalResult.case_id == case_id)
        .order_by(EvalResult.evaluated_at.desc())
        .all()
    )

    return EvalResultListResponse(items=results, total=len(results))
