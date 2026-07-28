"""
Generate a CI eval report by running the golden-case suite against the
mock LLM provider and comparing it to a checked-in baseline snapshot.

WHY the mock provider, not Gemini?
  CI has no real API budget, and MockLLMClient is deterministic — the
  same code always produces the same composite scores for the same
  golden cases. That determinism is what makes this a genuine regression
  gate: if a PR changes the embedding, judge, or scoring logic in a way
  that shifts these scores, this catches it.

  This is NOT a live production drift check — that's what
  GET /drift/{model_version} is for, which reads real eval_results
  history accumulated from actual provider traffic.

baseline_scores.json (repo root) is a committed snapshot of composite
scores from a known-good run of this same suite. Regenerate it after an
intentional, reviewed change to scoring/embedding/judge logic:

    python scripts/run_eval.py --write-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.drift import detect_drift_from_scores
from app.core.eval_service import run_eval as evaluate_case
from app.db.models.golden_case import GoldenCase
from app.db.session import get_db
from scripts.seed import SEED_CASES

BASELINE_PATH = Path(__file__).resolve().parent.parent / "baseline_scores.json"

# The golden suite has ~20 cases. This is a static snapshot-vs-snapshot
# comparison (not the 24h/7-day rolling production windows detect_drift()
# uses), so it needs a much lower sample floor than that endpoint's
# defaults (10 current / 30 baseline) to ever run the real statistics.
MIN_SAMPLES_FOR_STATS = 5


def run_suite(model_version: str, provider: str) -> list[float]:
    """Run every case in SEED_CASES against `provider`, returning composite scores."""
    scores: list[float] = []
    with get_db() as db:
        existing_ids = {
            prompt: case_id
            for prompt, case_id in db.query(GoldenCase.prompt, GoldenCase.case_id).all()
        }

        for data in SEED_CASES:
            case_id = existing_ids.get(data["prompt"])
            if case_id is not None:
                case = db.query(GoldenCase).filter(GoldenCase.case_id == case_id).one()
            else:
                case = GoldenCase(
                    case_id=uuid.uuid4(),
                    prompt=data["prompt"],
                    expected_topics=data.get("expected_topics", []),
                    safety_rules=data.get("safety_rules", []),
                    domain=data.get("domain"),
                    version_tag=data.get("version_tag"),
                )
                db.add(case)
                db.flush()

            result = evaluate_case(db, case, model_version=model_version, provider=provider)
            db.flush()
            if result.composite_score is not None:
                scores.append(round(float(result.composite_score), 6))

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument("--model-version", default="ci-baseline-mock")
    parser.add_argument(
        "--provider",
        default="mock",
        help="LLM provider to run the suite against (default: mock — deterministic, free, no API key needed)",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite baseline_scores.json with this run's scores instead of comparing against it.",
    )
    args = parser.parse_args()

    current_scores = run_suite(args.model_version, args.provider)
    if not current_scores:
        raise SystemExit("No composite scores produced — golden case suite or scoring pipeline is broken.")

    if args.write_baseline:
        BASELINE_PATH.write_text(json.dumps(current_scores, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(current_scores)} scores to {BASELINE_PATH}")
        return

    if BASELINE_PATH.exists():
        baseline_scores = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    else:
        print(f"No baseline file at {BASELINE_PATH} — treating this run as the new baseline.")
        baseline_scores = current_scores

    drift = detect_drift_from_scores(
        model_version=args.model_version,
        current_scores=current_scores,
        baseline_scores=baseline_scores,
        min_current_samples=min(MIN_SAMPLES_FOR_STATS, len(current_scores)),
        min_baseline_samples=min(MIN_SAMPLES_FOR_STATS, len(baseline_scores)),
    )

    composite_score = drift.current_mean if drift.current_mean is not None else 0.0
    baseline_score = drift.baseline_mean if drift.baseline_mean is not None else composite_score

    report = {
        "composite_score": composite_score,
        "baseline_score": baseline_score,
        "delta": round(composite_score - baseline_score, 3),
        "drift_detected": bool(drift.drift_detected),
        "p_value": drift.p_value,
        "effect_size": drift.effect_size,
        "sample_count": len(current_scores),
    }

    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
