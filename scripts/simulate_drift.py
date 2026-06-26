"""
scripts/simulate_drift.py

Simulates a real LLM evaluation pipeline producing live data.
Inserts new eval_results rows every few seconds via the Docker Postgres
so you can watch the Grafana gauges move.

Usage:
    # From project root, with venv activated:
    python scripts/simulate_drift.py

    # Custom options:
    python scripts/simulate_drift.py --model my-model --interval 3 --mode degrading
    python scripts/simulate_drift.py --mode stable
    python scripts/simulate_drift.py --mode recovery

Modes:
    degrading  → scores start good (~0.85) and slowly drop to ~0.50
    stable     → scores stay consistently good (~0.85), no drift
    recovery   → scores start bad (~0.50) and climb back to ~0.85
"""

from __future__ import annotations

import argparse
import random
import time
import uuid
from datetime import datetime, timezone

import psycopg2

# ── Config ────────────────────────────────────────────────────────────────────

# Port 5433 is the Docker Postgres exposed on your Mac.
# Port 5432 is taken by your local Homebrew Postgres — we avoid it.
DATABASE_URL = "postgresql://drift:drift@localhost:5433/driftscope"

# ── Helpers ───────────────────────────────────────────────────────────────────


def get_or_create_case(conn) -> str:
    """Return an existing case_id or create a dummy one for simulation."""
    with conn.cursor() as cur:
        cur.execute("SELECT case_id FROM golden_cases LIMIT 1")
        row = cur.fetchone()
        if row:
            return str(row[0])

        # Create a placeholder case so FK constraint is satisfied
        case_id = str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO golden_cases (case_id, prompt, created_at)
            VALUES (%s, %s, %s)
            """,
            (case_id, "[simulation] dummy golden case", datetime.now(timezone.utc)),
        )
        conn.commit()
        print(f"  Created dummy golden case: {case_id}")
        return case_id


def insert_eval(conn, case_id: str, model_version: str, composite: float, judge: float):
    """Insert one simulated eval_result row with current timestamp."""
    result_id = str(uuid.uuid4())
    # cosine_score is derived from composite:
    # composite = 0.6 * cosine + 0.4 * judge  →  cosine = (composite - 0.4*judge) / 0.6
    cosine = max(0.0, min(1.0, (composite - 0.4 * judge) / 0.6))

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_results
                (result_id, case_id, model_version,
                 response_text, cosine_score, judge_score, composite_score,
                 provider, evaluated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                result_id,
                case_id,
                model_version,
                f"[simulated response] composite={composite:.3f}",
                round(cosine, 4),
                round(judge, 4),
                round(composite, 4),
                "simulation",
                datetime.now(timezone.utc),
            ),
        )
    conn.commit()


def score_for_step(step: int, total_steps: int, mode: str) -> tuple[float, float]:
    """
    Return (composite_score, judge_score) for the current simulation step.

    degrading : high → low  (simulates a model going bad)
    stable    : stays high  (healthy model, no drift)
    recovery  : low → high  (simulates a fix being deployed)
    """
    noise = random.uniform(-0.03, 0.03)

    if mode == "degrading":
        # Linear decline from 0.88 down to 0.50 over all steps
        composite = 0.88 - (0.38 * step / total_steps) + noise
        judge = 0.90 - (0.40 * step / total_steps) + noise

    elif mode == "stable":
        composite = 0.85 + noise
        judge = 0.90 + noise

    else:  # recovery
        composite = 0.50 + (0.38 * step / total_steps) + noise
        judge = 0.55 + (0.40 * step / total_steps) + noise

    # Clamp to [0, 1]
    return max(0.0, min(1.0, composite)), max(0.0, min(1.0, judge))


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Simulate live LLM eval data for DriftScope")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model version label")
    parser.add_argument("--interval", type=float, default=4.0, help="Seconds between inserts")
    parser.add_argument("--steps", type=int, default=60, help="Total rows to insert")
    parser.add_argument(
        "--mode",
        choices=["degrading", "stable", "recovery"],
        default="degrading",
        help="Simulation mode",
    )
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════╗
║       DriftScope Live Simulation             ║
╠══════════════════════════════════════════════╣
║  Model   : {args.model:<32} ║
║  Mode    : {args.mode:<32} ║
║  Steps   : {args.steps:<32} ║
║  Interval: {args.interval:<32} ║
╚══════════════════════════════════════════════╝

  Watch the Grafana dashboard at http://localhost:3000
  Prometheus scrapes every 15s — gauges update shortly after each insert.

  Press Ctrl+C to stop early.
""")

    conn = psycopg2.connect(DATABASE_URL)
    case_id = get_or_create_case(conn)

    try:
        for step in range(args.steps):
            composite, judge = score_for_step(step, args.steps, args.mode)
            insert_eval(conn, case_id, args.model, composite, judge)

            bar_filled = int((step + 1) / args.steps * 30)
            bar = "█" * bar_filled + "░" * (30 - bar_filled)
            print(
                f"  [{bar}] step {step+1:>3}/{args.steps}  "
                f"composite={composite:.3f}  judge={judge:.3f}",
                end="\r",
            )

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n  Simulation stopped early.")
    finally:
        conn.close()

    print(f"\n\n  Done! Inserted {args.steps} rows for model '{args.model}'.")
    print("  Check http://localhost:3000 for updated gauges.\n")


if __name__ == "__main__":
    main()
