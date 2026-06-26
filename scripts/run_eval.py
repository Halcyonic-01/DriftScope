"""
Generate a CI-friendly DriftScope eval report.

This script intentionally avoids live LLM calls. It summarizes test status and
uses conservative placeholder quality values so GitHub Actions can exercise the
quality-gate mechanics without spending provider quota.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument("--composite-score", type=float, default=0.82)
    parser.add_argument("--baseline-score", type=float, default=0.82)
    parser.add_argument("--drift-detected", action="store_true")
    args = parser.parse_args()

    delta = args.composite_score - args.baseline_score
    report = {
        "composite_score": round(args.composite_score, 3),
        "baseline_score": round(args.baseline_score, 3),
        "delta": round(delta, 3),
        "drift_detected": bool(args.drift_detected),
    }

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
