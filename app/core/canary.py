"""
Provider-change canary using embedding centroid tracking.
"""

from __future__ import annotations

import logging
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Sequence
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from app.core.embeddings import get_model
from app.core.llm import get_client
from app.db.models.centroid_history import CentroidHistory
from app.db.models.golden_case import GoldenCase

logger = logging.getLogger(__name__)

DEFAULT_CANARY_THRESHOLD = 0.05
DEFAULT_BASELINE_DAYS = 7
# One request per case with no gap would breach a 15/min free-tier
# limit within seconds; 5s keeps a full 20-case run at 12/min.
DEFAULT_CANARY_DELAY_SECONDS = 5.0


@dataclass(frozen=True)
class CanaryResult:
    provider: str
    drift_score: float
    alert_sent: bool
    response_count: int
    history_id: str


def compute_centroid(responses: Sequence[str]) -> np.ndarray:
    """Return the mean normalized embedding for a set of provider responses."""
    cleaned = [response.strip() for response in responses if response and response.strip()]
    if not cleaned:
        raise ValueError("Cannot compute centroid without at least one response")

    embeddings = get_model().encode(cleaned, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=float).mean(axis=0)


def centroid_drift(current: Sequence[float], previous: Sequence[float]) -> float:
    """Cosine distance between two centroids. 0.0 is identical."""
    current_arr = np.asarray(current, dtype=float).flatten()
    previous_arr = np.asarray(previous, dtype=float).flatten()
    denom = np.linalg.norm(current_arr) * np.linalg.norm(previous_arr)
    if denom == 0:
        return 0.0
    similarity = float(np.dot(current_arr, previous_arr) / denom)
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def run_canary(
    db: Session,
    provider: str,
    golden_case_ids: Sequence[UUID | str] | None = None,
    alert_threshold: float = DEFAULT_CANARY_THRESHOLD,
    baseline_days: int = DEFAULT_BASELINE_DAYS,
    case_limit: int | None = None,
    delay_seconds: float = DEFAULT_CANARY_DELAY_SECONDS,
) -> CanaryResult:
    """
    Run the canary prompt set against a provider and store a centroid snapshot.

    delay_seconds paces the calls. The canary issues one request per case
    with nothing between them, so an unpaced 20-case run fires 20 requests
    in seconds and breaches a per-minute provider limit immediately.
    """
    provider = provider.lower().strip()
    cases = _load_cases(db, golden_case_ids, case_limit=case_limit)
    llm = get_client(provider)

    responses = []
    for index, case in enumerate(cases, start=1):
        if delay_seconds and index > 1:
            time.sleep(delay_seconds)
        logger.info(
            "Canary provider=%s case=%s/%s id=%s",
            provider,
            index,
            len(cases),
            case.case_id,
        )
        responses.append(llm.complete(case.prompt).text)
    current_centroid = compute_centroid(responses)
    # Compare only against snapshots built from the same number of
    # prompts. A centroid is the mean of its prompt set, so one computed
    # from 5 cases and one from 20 differ because the inputs differ —
    # averaging them together reports large "drift" that says nothing
    # about the model.
    previous_centroid = get_rolling_centroid(
        db,
        provider,
        baseline_days=baseline_days,
        case_count=len(responses),
    )
    drift_score = (
        centroid_drift(current_centroid, previous_centroid)
        if previous_centroid is not None
        else 0.0
    )

    history = CentroidHistory(
        provider=provider,
        centroid=[float(value) for value in current_centroid.tolist()],
        drift_score=round(float(drift_score), 6),
        case_count=len(responses),
    )
    db.add(history)
    db.flush()

    alert_sent = False
    if drift_score > alert_threshold:
        alert_sent = send_configured_email_alert(provider, drift_score, alert_threshold)

    return CanaryResult(
        provider=provider,
        drift_score=round(float(drift_score), 4),
        alert_sent=alert_sent,
        response_count=len(responses),
        history_id=str(history.id),
    )


def get_rolling_centroid(
    db: Session,
    provider: str,
    baseline_days: int = DEFAULT_BASELINE_DAYS,
    case_count: int | None = None,
) -> np.ndarray | None:
    """
    Mean centroid over the baseline window for a provider.

    When case_count is given, only snapshots built from that many prompts
    are included. Comparing across prompt-set sizes measures the change in
    inputs rather than the change in the model.
    """
    since = datetime.now(timezone.utc) - timedelta(days=baseline_days)
    query = (
        db.query(CentroidHistory.centroid)
        .filter(CentroidHistory.provider == provider)
        .filter(CentroidHistory.recorded_at >= since)
    )
    if case_count is not None:
        query = query.filter(CentroidHistory.case_count == case_count)

    rows = query.order_by(CentroidHistory.recorded_at.desc()).all()
    if not rows:
        return None
    return np.asarray([row[0] for row in rows], dtype=float).mean(axis=0)


def send_configured_email_alert(
    provider: str,
    drift_score: float,
    threshold: float = DEFAULT_CANARY_THRESHOLD,
) -> bool:
    """Send an alert if SMTP settings are complete; return whether one was sent."""
    from app.core.config import settings

    if not all(
        [
            settings.alert_email,
            settings.smtp_host,
            settings.smtp_port,
            settings.smtp_user,
            settings.smtp_password,
        ]
    ):
        logger.warning("Canary drift detected but SMTP settings are incomplete")
        return False

    send_email_alert(
        provider=provider,
        drift_score=drift_score,
        threshold=threshold,
        to_email=settings.alert_email,
        smtp_host=settings.smtp_host,
        smtp_port=settings.smtp_port,
        smtp_user=settings.smtp_user,
        smtp_password=settings.smtp_password,
    )
    return True


def send_email_alert(
    provider: str,
    drift_score: float,
    threshold: float,
    to_email: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
) -> None:
    detected_at = datetime.now(timezone.utc).isoformat()
    subject = f"DriftScope Canary Alert - {provider} drift detected"
    body = (
        "DriftScope Canary Alert\n"
        "=======================\n"
        f"Provider       : {provider}\n"
        f"Centroid Drift : {drift_score:.4f} (threshold: {threshold:.4f})\n"
        f"Detected at    : {detected_at}\n\n"
        "A silent model update may have occurred. Review the DriftScope dashboard."
    )

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, msg.as_string())


def _load_cases(
    db: Session,
    golden_case_ids: Sequence[UUID | str] | None,
    case_limit: int | None = None,
) -> list[GoldenCase]:
    query = db.query(GoldenCase)
    if golden_case_ids:
        ids = [case_id if isinstance(case_id, UUID) else UUID(str(case_id)) for case_id in golden_case_ids]
        query = query.filter(GoldenCase.case_id.in_(ids))

    query = query.order_by(GoldenCase.created_at.asc())
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be at least 1")
        query = query.limit(case_limit)

    cases = query.all()
    if not cases:
        raise ValueError("No golden cases found for canary run")
    return cases
