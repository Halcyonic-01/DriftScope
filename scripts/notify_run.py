"""
Email a summary of a scheduled workflow run.

GitHub only emails you when a workflow *fails*, and only for the default
branch. A run that succeeds — or one that succeeds while quietly
evaluating zero cases — never reaches your inbox, so "the monitor is
still alive" is unverifiable without opening the Actions tab. This sends
that confirmation.

Called at the end of a workflow with `if: always()`, so the outcome is
reported whether the job passed or failed:

    python scripts/notify_run.py \
        --job monitor \
        --status "${{ job.status }}" \
        --summary-file monitor_summary.json \
        --run-url "$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"

Exits 0 even when the mail fails. Notification is not the job — a dead
SMTP server must not turn a green monitoring run red.
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

from app.core.notify import send_email, smtp_is_configured

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_STATUS_ICON = {"success": "OK", "failure": "FAILED", "cancelled": "CANCELLED"}


def _read_summary(path: str | None) -> str:
    """Render the job's own JSON summary, if it produced one."""
    if not path:
        return ""
    file = Path(path)
    if not file.exists():
        return "(no summary file was produced - the job likely failed early)\n"

    text = file.read_text(encoding="utf-8").strip()
    if not text:
        return "(summary file was empty)\n"

    # run_canary.py writes JSONL (one record per provider); run_monitor.py
    # writes a single object. Handle both by parsing line by line.
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            lines.append(raw)
            continue
        for key, value in sorted(data.items()):
            lines.append(f"  {key:<22}: {value}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Job name, e.g. monitor or canary.")
    parser.add_argument(
        "--status",
        default="unknown",
        help="Outcome from GitHub's job.status: success, failure, or cancelled.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="JSON or JSONL file the job wrote, included verbatim in the body.",
    )
    parser.add_argument("--run-url", default=None, help="Link back to the Actions run.")
    args = parser.parse_args()

    if not smtp_is_configured():
        # Not an error: local runs and forks legitimately have no SMTP.
        logger.warning(
            "Skipping run notification for job=%s: SMTP settings are incomplete. "
            "Set ALERT_EMAIL, SMTP_USER and SMTP_PASSWORD to receive these.",
            args.job,
        )
        return

    status = args.status.lower()
    icon = _STATUS_ICON.get(status, status.upper() or "UNKNOWN")
    subject = f"DriftScope {args.job} run {icon}"

    body_parts = [
        f"DriftScope scheduled job: {args.job}",
        "=" * 40,
        f"Status     : {icon}",
        f"Finished   : {datetime.now(timezone.utc).isoformat()}",
    ]
    if args.run_url:
        body_parts.append(f"Run        : {args.run_url}")
    body_parts.append("")
    body_parts.append("Result")
    body_parts.append("-" * 40)
    body_parts.append(_read_summary(args.summary_file) or "(no summary requested)")

    try:
        send_email(subject=subject, body="\n".join(body_parts))
        logger.info("Sent run notification for job=%s status=%s", args.job, status)
    except Exception as exc:
        # Deliberately swallowed — see the module docstring.
        logger.error("Could not send run notification for job=%s: %s", args.job, exc)


if __name__ == "__main__":
    main()
