"""
Run the golden-case suite against a live provider and record the results.

This is the job that actually keeps DriftScope's monitoring alive. The
/metrics gauges average over a rolling 24h window and detect_drift()
needs >=10 results in 24h plus >=30 in the preceding 7 days, so if
nothing runs on a schedule the dashboard goes blank and drift detection
reports insufficient_data forever.

Unlike scripts/run_eval.py (the PR quality gate, which runs against the
deterministic mock provider and compares to a committed baseline), this
hits the real provider so the recorded scores reflect live model quality.

    python scripts/run_monitor.py --provider gemini --model-version gemini-3.5-flash-lite

Each case is evaluated independently: one provider error (a quota block,
a safety refusal) is logged and skipped rather than aborting the run and
leaving a partial hole in the time series.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.eval_service import run_eval
from app.db.models.golden_case import GoldenCase
from app.db.models.job_run import JobRun
from app.db.session import get_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument(
        "--model-version",
        default=None,
        help="Label recorded on each result. Defaults to the provider name.",
    )
    parser.add_argument("--domain", default=None, help="Only run cases in this domain.")
    parser.add_argument(
        "--summary-file",
        default=None,
        help=(
            "Also write the run summary here as JSON. scripts/notify_run.py "
            "reads this to build the confirmation email."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap how many cases to run.")
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.5,
        help="Exit non-zero if this fraction or more of cases fail (default 0.5).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=8.0,
        help=(
            "Seconds to wait between cases (default 8.0). A case costs 2 API "
            "calls: one generation plus one batched judge call covering every "
            "rule. gemini-3.5-flash-lite allows 15 requests/minute, so 8s "
            "between cases keeps a run comfortably inside that. Raise it if "
            "you switch to a model with a lower RPM (the standard Flash "
            "models allow only 5/min and 20/day). Pass 0 to disable."
        ),
    )
    args = parser.parse_args()

    model_version = args.model_version or args.provider
    succeeded = 0
    failed = 0
    last_error: str | None = None
    scores: list[float] = []

    with get_db() as db:
        query = db.query(GoldenCase)
        if args.domain:
            query = query.filter(GoldenCase.domain == args.domain)
        query = query.order_by(GoldenCase.created_at.asc())
        if args.limit is not None:
            query = query.limit(args.limit)
        # Detach the ids now; each case is re-loaded in its own session
        # below, so nothing here outlives this connection.
        case_ids = [case.case_id for case in query.all()]

    if not case_ids:
        raise SystemExit(
            "No golden cases found. Seed them first: python scripts/seed.py"
        )

    logger.info(
        "Evaluating %d cases against provider=%s as model_version=%s",
        len(case_ids), args.provider, model_version,
    )

    # A session per case, rather than one held open for the whole run.
    # The run spends most of its time sleeping between provider calls, and
    # a hosted database (Neon's free tier especially) drops connections
    # that sit idle -- which surfaced as
    # "SSL connection has been closed unexpectedly" partway through a run
    # and silently lost every remaining case. Taking a fresh connection
    # per case lets pool_pre_ping validate it before use.
    for index, case_id in enumerate(case_ids, start=1):
        if args.delay and index > 1:
            time.sleep(args.delay)
        try:
            with get_db() as db:
                case = db.query(GoldenCase).filter(GoldenCase.case_id == case_id).one()
                result = run_eval(
                    db=db,
                    case=case,
                    model_version=model_version,
                    provider=args.provider,
                )
                composite = result.composite_score
                # get_db() commits on exit, so each case is durable before
                # the next one starts -- a partial time series is far more
                # useful than an empty one.
            succeeded += 1
            if composite is not None:
                scores.append(float(composite))
            logger.info(
                "[%d/%d] case=%s composite=%s",
                index, len(case_ids), case_id,
                f"{composite:.4f}" if composite is not None else "n/a",
            )
        except Exception as exc:
            failed += 1
            last_error = str(exc)[:500]
            logger.error("[%d/%d] case=%s failed: %s", index, len(case_ids), case_id, exc)

    # Always record the run, especially when everything failed -- that's
    # precisely the case that otherwise leaves no trace at all and looks
    # identical to "the job never ran".
    with get_db() as db:
        db.add(
            JobRun(
                job="monitor",
                provider=args.provider,
                model_version=model_version,
                cases_total=succeeded + failed,
                cases_succeeded=succeeded,
                cases_failed=failed,
                last_error=last_error,
            )
        )

    total = succeeded + failed
    summary = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model_version": model_version,
        "cases_total": total,
        "cases_succeeded": succeeded,
        "cases_failed": failed,
        "mean_composite_score": round(sum(scores) / len(scores), 4) if scores else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    # Written before the threshold check below, so a run that fails the
    # threshold still reports what it found instead of emailing a blank.
    if args.summary_file:
        Path(args.summary_file).write_text(
            json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
        )

    if total and (failed / total) >= args.fail_threshold:
        raise SystemExit(
            f"{failed}/{total} cases failed (threshold {args.fail_threshold:.0%}) - failing the run."
        )


if __name__ == "__main__":
    main()
