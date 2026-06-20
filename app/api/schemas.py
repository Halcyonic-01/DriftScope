"""
app/api/schemas.py

Pydantic schemas for request and response validation.

WHY separate schemas from DB models?
  Your DB model (SQLAlchemy) describes HOW data is stored in PostgreSQL.
  Your API schema (Pydantic) describes WHAT the HTTP API accepts and returns.
  They are different concerns:
    - DB model has internal fields (created_at, foreign keys, etc.)
    - API schema only exposes what the caller needs to see or send
  Keeping them separate means you can change one without breaking the other.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Request Schemas (what the caller sends) ──────────────────────────────────

class CreateCaseRequest(BaseModel):
    """Body for POST /cases — create a new golden test case."""
    prompt: str = Field(
        ...,
        min_length=10,
        description="The prompt that will be sent to the LLM under test.",
        examples=["Explain the side effects of ibuprofen in simple terms."],
    )
    expected_topics: List[str] = Field(
        default=[],
        description="Topics the response MUST cover to pass the eval.",
        examples=[["side effects", "dosage", "contraindications"]],
    )
    safety_rules: List[str] = Field(
        default=[],
        description="Rules the response must NOT violate.",
        examples=[["Must not recommend dosage above 400mg", "Must advise consulting a doctor"]],
    )
    version_tag: Optional[str] = Field(
        default=None,
        description="Model/prompt version this case belongs to (e.g. 'v1.2-flash').",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain category (e.g. 'medical', 'legal', 'finance', 'general').",
    )


class RunEvalRequest(BaseModel):
    """Body for POST /cases/{case_id}/run — execute an eval against an LLM."""
    model_version: str = Field(
        ...,
        description="Label for which LLM version is being evaluated (e.g. 'gemini-2.5-flash-v1').",
        examples=["gemini-2.5-flash-v1"],
    )
    provider: str = Field(
        default="mock",
        description="Which LLM to use: 'gemini', 'ollama', or 'mock'.",
    )


# ── Response Schemas (what the API returns) ───────────────────────────────────

class CaseResponse(BaseModel):
    """Returned for a single golden case."""
    case_id: UUID
    prompt: str
    expected_topics: List[str]
    safety_rules: List[str]
    version_tag: Optional[str]
    domain: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class CaseListResponse(BaseModel):
    """Returned for GET /cases — paginated list."""
    items: List[CaseResponse]
    total: int
    page: int
    page_size: int


class EvalResultResponse(BaseModel):
    """Returned after running an eval."""
    result_id: UUID
    case_id: UUID
    model_version: str
    response_text: Optional[str]
    cosine_score: Optional[float]
    judge_score: Optional[float]
    composite_score: Optional[float]
    judge_reason: Optional[str]
    provider: Optional[str]
    evaluated_at: datetime

    model_config = {"from_attributes": True}


class EvalResultListResponse(BaseModel):
    """Returned for GET /cases/{case_id}/results."""
    items: List[EvalResultResponse]
    total: int


class ModelReportResponse(BaseModel):
    """Returned for GET /reports/{model_version}."""
    model_version: str
    total_runs: int
    avg_composite_score: Optional[float]
    avg_cosine_score: Optional[float]
    judge_pass_rate: Optional[float]
    judge_invocation_rate: float
    evaluated_from: Optional[datetime]
    evaluated_to: Optional[datetime]
