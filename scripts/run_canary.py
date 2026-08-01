"""
Run the Phase 4 provider-change canary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.canary import (
    DEFAULT_CANARY_DELAY_SECONDS,
    DEFAULT_CANARY_THRESHOLD,
    run_canary,
)
from app.db.session import get_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    # Only gemini by default: ollama needs a local `ollama serve` and a
    # pulled model, so including it by default made the script fail for
    # anyone who hasn't set that up. Add it explicitly when you have.
    parser.add_argument("--providers", nargs="+", default=["gemini"])
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CANARY_THRESHOLD)
    parser.add_argument("--baseline-days", type=int, default=7)
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_CANARY_DELAY_SECONDS,
        help=(
            "Seconds between cases (default %(default)s). The canary makes one\n"
            "call per case with no natural gap, so an unpaced 20-case run\n"
            "breaches a 15/min free-tier limit immediately. Pass 0 to disable."
        ),
    )
    parser.add_argument("--output-jsonl", default="canary_log.jsonl")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    # Each provider gets its own get_db() session/transaction. Sharing one
    # transaction across all providers meant a single provider failing
    # (e.g. Gemini quota exhausted) would roll back every other provider's
    # already-successful centroid_history write too, since get_db() only
    # commits once at the very end of the `with` block.
    records = []
    had_failure = False

    for provider in args.providers:
        try:
            with get_db() as db:
                result = run_canary(
                    db=db,
                    provider=provider,
                    golden_case_ids=args.case_ids,
                    alert_threshold=args.threshold,
                    baseline_days=args.baseline_days,
                    case_limit=args.limit,
                    delay_seconds=args.delay,
                )
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "provider": result.provider,
                "drift_score": result.drift_score,
                "alert_sent": result.alert_sent,
                "response_count": result.response_count,
                "history_id": result.history_id,
            }
        except Exception as exc:
            logger.error("Canary run failed for provider=%s: %s", provider, exc)
            had_failure = True
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "error": str(exc),
            }

        records.append(record)
        print(json.dumps(record, sort_keys=True))

    if args.output_jsonl:
        path = Path(args.output_jsonl)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    if had_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
