<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/DriftScope-LLM%20Quality%20Monitor-6366f1?style=for-the-badge&labelColor=0f0f0f">
  <img alt="DriftScope" src="https://img.shields.io/badge/DriftScope-LLM%20Quality%20Monitor-6366f1?style=for-the-badge">
</picture>

# 🔭 DriftScope

> **A full-stack LLM quality monitoring platform built from scratch.**
> Designed to detect silent model regressions, run statistically-grounded drift detection, and catch provider-side model swaps before users do.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20Flash%20Lite-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)

---

## 📖 Table of Contents

- [Why DriftScope](#-why-driftscope)
- [What DriftScope Does](#-what-driftscope-does)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Free LLM Choice](#-free-llm-choice)
- [Phase Implementation Plan](#-phase-implementation-plan)
  - [Phase 1 — Foundation](#phase-1--foundation)
  - [Phase 2 — Intelligence Layer](#phase-2--intelligence-layer)
  - [Phase 3 — Drift Detection & DevOps](#phase-3--drift-detection--devops)
  - [Phase 4 — Provider-Change Canary & Empirical Study](#phase-4--provider-change-canary--empirical-study)
- [Research Base](#-research-base)
- [Quick-Start Checklist](#-quick-start-checklist)
- [Getting Started](#-getting-started)
- [Keeping Monitoring Alive](#-keeping-monitoring-alive)
- [Running Tests](#-running-tests)

---

## 🚨 Why DriftScope

The underlying pain is real even if the tooling space is crowded. Enterprises spend approximately **$14,200 per employee annually** dealing with LLM hallucinations and silent quality regressions — roughly 4.3 hours per worker per week on fact-checking and error correction. Teams that ship without continuous evaluation typically discover regressions from **customer complaints days after the fact**.

> *"There is no guarantee that the system named GPT-4o at 16:18 will be the same system at 18:16."*  
> — Murphy & Underwood, ACM Queue 2025

The specific gap DriftScope addresses: **no existing open-source tool detects when a provider silently updates their model** — your application’s behaviour changes with no API version bump, no changelog, no warning. DriftScope is built to solve exactly this.

---

## 🏗️ What DriftScope Does

DriftScope is a **five-module system**, each independently useful, combined into one platform:

| # | Module | Description | Novel? |
|---|--------|-------------|--------|
| 1 | **Golden Dataset Store** | Versioned PostgreSQL store of `(prompt, expected_behavior)` behavioural contracts | — |
| 2 | **Multi-Signal Scorer** | Embedding cosine sim + LLM-as-judge + composite weighted score | — |
| 3 | **Statistical Drift Detector** | Mann-Whitney U test on rolling score distributions | ✅ Novel |
| 4 | **CI/CD Quality Gate** | GitHub Actions that blocks PRs on >5% composite score drop | — |
| 5 | **Provider-Change Canary** | Nightly SBERT centroid tracking to catch silent model updates | ✅ Novel |

---

## 🧭 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DriftScope Platform                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  │   FastAPI    │───▶│  Multi-Signal    │───▶│  Statistical      │ │
│  │   REST API   │    │  Scorer          │    │  Drift Detector   │ │
│  └──────┬───────┘    │  · Cosine Sim    │    │  · Mann-Whitney U │ │
│         │            │  · LLM Judge     │    │  · Cohen's d      │ │
│         ▼            │  · Composite     │    │  · 24h vs 7d      │ │
│  ┌──────────────┐    └────────┬─────────┘    └────────┬──────────┘ │
│  │  PostgreSQL  │◀───────────┘                        │            │
│  │  · golden_   │                                     │            │
│  │    cases     │◀────────────────────────────────────┘            │
│  │  · eval_     │                                                   │
│  │    results   │    ┌──────────────────┐    ┌───────────────────┐ │
│  │  · centroid_ │◀───│  Nightly Canary  │    │  GitHub Actions   │ │
│  │    history   │    │  · Gemini Flash  │    │  CI/CD Gate       │ │
│  └──────────────┘    │  · Ollama (local)│    │  · PR Comments    │ │
│                      └──────────────────┘    │  · Merge Blocking │ │
│                                              └───────────────────┘ │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  │  Prometheus  │───▶│    Grafana       │    │  Email Notifier   │ │
│  │  /metrics    │    │    Dashboard     │    │  · Drift alerts   │ │
│  └──────────────┘    └──────────────────┘    │  · Run summaries  │ │
│                                              └───────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Database Schema

```sql
-- Behavioural contracts — NOT expected strings
CREATE TABLE golden_cases (
    case_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt         TEXT NOT NULL,
    expected_topics TEXT[],       -- themes the response should cover
    safety_rules   TEXT[],        -- natural-language rules for LLM judge
    version_tag    VARCHAR(50),   -- e.g. "v1.2-gemini-flash"
    domain         VARCHAR(50),   -- e.g. "medical", "legal", "finance"
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- Every eval run stored here — enables rolling window queries
CREATE TABLE eval_results (
    result_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id         UUID REFERENCES golden_cases(case_id),
    model_version   VARCHAR(100),
    response_text   TEXT,
    cosine_score    FLOAT,
    judge_score     FLOAT,
    composite_score FLOAT,
    provider        VARCHAR(50),  -- "gemini", "ollama", "local"
    evaluated_at    TIMESTAMPTZ DEFAULT now()
);

-- Phase 4: centroid history for canary
CREATE TABLE centroid_history (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider        VARCHAR(50),
    centroid        FLOAT[],      -- embedding centroid vector
    drift_score     FLOAT,
    case_count      INTEGER,      -- how many prompts built this centroid
    recorded_at     TIMESTAMPTZ DEFAULT now()
);

-- Outcomes of scheduled runs, so a job that failed every case is
-- distinguishable from a job that never ran
CREATE TABLE job_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job             VARCHAR(50),  -- "monitor", "canary"
    provider        VARCHAR(50),
    model_version   VARCHAR(100),
    cases_total     INTEGER,
    cases_succeeded INTEGER,
    cases_failed    INTEGER,
    last_error      TEXT,
    finished_at     TIMESTAMPTZ DEFAULT now()
);
```

`centroid_history.case_count` exists because a centroid built from 20 prompts isn't comparable to one built from 5 — the baseline query filters to snapshots of the same size, so drift measures the model changing rather than the prompt set changing.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| API Framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + Alembic (migrations) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), PyTorch |
| Statistics | scipy (`scipy.stats.mannwhitneyu`) |
| CI/CD | GitHub Actions |
| Observability | Prometheus + Grafana |
| Infrastructure | Docker Compose |
| LLM (Judge + Canary) | **Google Gemini Flash Lite** (free tier), Ollama (local fallback) |

---

## 🤖 Free LLM Choice

**Google Gemini Flash Lite** (`gemini-3.5-flash-lite`, set by `GEMINI_MODEL`) is used as the LLM for both the judge and the canary runs. Here’s why it’s the best free option for this project:

| Criterion | Gemini Flash Lite |
|-----------|-----------------|
| **Cost** | Free tier via [Google AI Studio](https://aistudio.google.com) — 15 RPM, 500 requests/day |
| **Structured output** | Native JSON mode — critical for the LLM-as-judge `{pass, reason}` schema |
| **Speed** | Sub-second latency — fast enough for nightly canary runs |
| **Context window** | 1M tokens — handles long responses without truncation |
| **Python SDK** | `google-generativeai` — simple, well-documented |
| **Quality** | Judges accurately, correctly failing responses that violate an explicit safety rule |

The **Lite** tier is a deliberate choice over standard Flash. On the free tier the standard Flash models allow only 20 requests/day, which caps a run at 10 cases and can't sustain continuous monitoring. Flash Lite's 500/day is 25× the budget at the same judging quality.

The model is also pinned to a concrete ID rather than a `-latest` alias. DriftScope exists to detect when a provider silently changes the model behind a fixed name; an alias that intentionally moves would manufacture drift signals and make the canary meaningless.

```bash
pip install google-generativeai
```

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-3.5-flash-lite")

response = model.generate_content(
    prompt,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json"
    )
)
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com).

---

## 📅 Phase Implementation Plan

---

### Phase 1 — Foundation

**Goal:** Stand up the core data layer, embedding utilities, and REST API skeleton. By the end of this phase, golden test cases can be stored and a basic cosine-similarity eval can be run against a live model.

#### 1.1 Environment Setup

```bash
pip install sentence-transformers fastapi uvicorn psycopg2-binary alembic \
            scipy numpy pytest httpx python-dotenv google-generativeai
```

The `.env.example` template:
```
DATABASE_URL=postgresql://user:pass@localhost:5432/driftscope
GEMINI_API_KEY=AIza...
```

#### 1.2 PostgreSQL Schema + Alembic Migration

- Initialise Alembic: `alembic init alembic/`
- Create the `golden_cases` and `eval_results` tables (see schema above)
- Run the initial migration: `alembic upgrade head`
- Write a seed script that inserts 20+ golden test cases for the chosen domain (e.g. medical Q&A, legal summarisation, customer support)

**Key design decision:** DriftScope stores `expected_topics` (what the model should cover) and `safety_rules` (natural-language rules for the LLM judge), **not** verbatim expected strings. This makes evals robust to the non-deterministic nature of LLM outputs.

#### 1.3 Embedding Utilities

Two core utility functions are implemented with full pytest unit tests:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # free, runs locally, ~5ms/call

def embed(text: str) -> np.ndarray:
    """Return a unit-normalised embedding vector."""
    return model.encode(text, normalize_embeddings=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised vectors."""
    return float(np.dot(a, b))
```

> **Why `all-MiniLM-L6-v2`?**  
> 384-dimensional, 22M params, runs in ~5ms on CPU. arXiv:2602.11165 shows models achieve >99% cosine similarity with gold references despite <8% BLEU-1 overlap — embeddings capture *meaning*, lexical metrics don't.

**Tests:**
- `test_embed_returns_unit_vector()`
- `test_cosine_sim_identical_texts()`
- `test_cosine_sim_orthogonal_texts()`
- `test_cosine_sim_threshold_at_0_80()`

#### 1.4 Unified LLM Client (Factory Pattern)

```python
class LLMClient:
    def __init__(self, provider: str):  # "gemini" | "local"
        ...
    def complete(self, prompt: str, schema: dict | None = None) -> dict:
        ...

def get_client(provider: str) -> LLMClient:
    """Factory — swap providers without touching business logic."""
    ...
```

**Gemini Flash Lite** is wired as the primary provider, with **Ollama** (local) as the fallback. The factory pattern means providers can be swapped without touching any business logic.

#### 1.5 FastAPI Endpoints

```
POST /cases                   → create a golden test case
GET  /cases                   → list all cases (paginated)
GET  /cases/{case_id}         → get single case
POST /cases/{case_id}/run     → run eval: call LLM, compute cosine score, store result
GET  /cases/{case_id}/results → history of eval results for a case
```

**Deliverable checklist:**
- [ ] `pip install` command works in a fresh virtualenv
- [ ] `alembic upgrade head` creates both tables cleanly
- [ ] `POST /cases` stores a case and returns its UUID
- [ ] `POST /cases/{id}/run` calls Gemini, embeds the response, stores cosine score
- [ ] Pytest suite passes for embedding utilities

---

### Phase 2 — Intelligence Layer

**Goal:** Add the LLM-as-judge, cost guard, composite scoring, and aggregated reporting. By end of this phase an 80%+ covered integration test suite with mocked LLM responses should be complete.

#### 2.1 LLM-as-Judge (Gemini Flash Lite)

A structured rubric prompt is implemented that returns `{pass: bool, reason: str}`:

```python
import google.generativeai as genai
import json

model = genai.GenerativeModel("gemini-3.5-flash-lite")

def judge_response(response: str, rule: str) -> dict:
    prompt = f"""You are a strict evaluator. Answer in JSON only.

Rule: {rule}
Response to evaluate: {response}

Does the response satisfy the rule?
Respond with: {{"pass": true/false, "reason": "one sentence explanation"}}"""

    result = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json")
    )
    return json.loads(result.text)
```

**Known biases I'll mitigate (from arXiv:2411.15594):**
- **Position bias:** Randomise rubric order when comparing multiple responses
- **Verbosity bias:** Keep rubrics length-agnostic — longer responses shouldn't automatically score higher
- **Self-enhancement bias:** Use the same model class for both judge and evaluee only when unavoidable

#### 2.2 Cost Guard

Even with the free tier, judge invocations should be deliberate. The judge is only called when:
1. `cosine_score < 0.65` (borderline semantic match), **OR**
2. The case has `"safety"` in its tags

```python
def should_invoke_judge(cosine_score: float, case_tags: list[str]) -> bool:
    return cosine_score < 0.65 or "safety" in case_tags
```

The `0.65` threshold is calibrated for `all-MiniLM-L6-v2` against natural LLM
answers. In practice, good long-form answers often land around `0.65–0.75`
because they include useful details beyond the compact reference contract. This
keeps the judge as a cost guard fallback instead of invoking it for most good
responses.

#### 2.3 Composite Weighted Score

```python
def composite_score(
    cosine: float,
    judge: float | None,
    w1: float = 0.6,
    w2: float = 0.4
) -> float:
    """
    Combine cosine similarity and judge pass rate into one number.
    Weights are configurable per deployment domain.
    Medical/legal → higher w2; general assistant → higher w1.
    """
    if judge is None:   # judge was skipped by cost guard
        return cosine
    return (w1 * cosine) + (w2 * judge)
```

`cosine_score`, `judge_score`, and `composite_score` are stored separately in `eval_results` for full auditability.

#### 2.4 Reporting Endpoint

```
GET /reports/{model_version}
```

Returns aggregated stats for a model version:
```json
{
  "model_version": "v1.2-gemini-flash",
  "total_runs": 342,
  "avg_composite_score": 0.871,
  "avg_cosine_score": 0.903,
  "judge_pass_rate": 0.84,
  "judge_invocation_rate": 0.13,
  "evaluated_from": "2026-06-01T00:00:00Z",
  "evaluated_to": "2026-06-19T00:00:00Z"
}
```

#### 2.5 Integration Tests (80%+ Coverage Target)

I'll use `pytest` + `httpx.AsyncClient` + `unittest.mock` to mock LLM responses so tests run fast and free:

```python
@pytest.mark.asyncio
async def test_run_eval_stores_composite_score(mock_llm_client, async_client, db_session):
    # Arrange: seed one golden case
    case = await create_test_case(db_session, domain="medical")
    mock_llm_client.complete.return_value = "Patient should rest and hydrate."

    # Act
    resp = await async_client.post(f"/cases/{case.case_id}/run",
                                   json={"provider": "gemini", "model_version": "v1.2-gemini-flash"})

    # Assert
    assert resp.status_code == 200
    result = resp.json()
    assert 0.0 <= result["composite_score"] <= 1.0
    assert "judge_score" in result
```

**Deliverable checklist:**
- [ ] LLM judge returns structured `{pass, reason}` for every invocation
- [ ] Cost guard correctly skips judge on high-cosine, non-safety cases
- [ ] `GET /reports/{model_version}` returns correct aggregates from DB
- [ ] Integration test suite: ≥80% line coverage (`pytest --cov`)
- [ ] Judge reason strings stored in `eval_results` for debugging

---

### Phase 3 — Drift Detection & DevOps

**Goal:** Add statistically-grounded drift detection, wire it into a GitHub Actions CI gate, and launch the full observability stack. By the end of this phase, a single `docker compose up` should spin the entire platform locally.

#### 3.1 Mann-Whitney Drift Detector

This is the **novel core** of DriftScope. Rather than comparing single scores, it compares *distributions* — this is what makes the drift detection defensible and not just vibes-based thresholds.

```python
from scipy.stats import mannwhitneyu
import numpy as np

def detect_drift(db_session, model_version: str) -> dict:
    # Pull rolling windows from eval_results
    today    = get_scores(db_session, model_version, hours=24)
    baseline = get_scores(db_session, model_version, days=7)

    if len(today) < 10 or len(baseline) < 30:
        return {"status": "insufficient_data"}

    stat, p_value = mannwhitneyu(today, baseline, alternative="less")

    # Cohen's d for practical effect size
    pooled_std = np.sqrt((np.std(today)**2 + np.std(baseline)**2) / 2)
    effect_size = (np.mean(baseline) - np.mean(today)) / (pooled_std + 1e-9)

    drift_detected = (p_value < 0.05) and (effect_size > 0.1)

    return {
        "drift_detected": drift_detected,
        "p_value":        round(p_value, 4),
        "effect_size":    round(effect_size, 3),
        "today_mean":     round(np.mean(today), 3),
        "baseline_mean":  round(np.mean(baseline), 3),
    }
```

**Why Mann-Whitney U, not a t-test?**  
Cosine score distributions are rarely Gaussian. Mann-Whitney U is non-parametric — no normality assumption required. Cohen's d adds *practical* significance on top of *statistical* significance, preventing false alerts from tiny real-world differences that happen to be statistically significant.

**API endpoint:**
```
GET /drift/{model_version}
```
Response:
```json
{
  "model_version": "v1.2-gemini-flash",
  "drift_detected": true,
  "p_value": 0.0231,
  "effect_size": 0.412,
  "today_mean": 0.801,
  "baseline_mean": 0.873
}
```

#### 3.2 GitHub Actions CI/CD Gate

A `.github/workflows/eval.yml` CI workflow is created:

```yaml
name: DriftScope Eval Gate

on:
  pull_request:
    branches: [main]

jobs:
  eval:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: driftscope_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run migrations
        run: alembic upgrade head
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/driftscope_test

      - name: Run eval suite
        id: eval
        run: |
          python scripts/run_eval.py --output eval_report.json
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/driftscope_test
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}

      - name: Post PR comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('eval_report.json'));
            const body = `## 🔭 DriftScope Eval Report
            | Metric | Value |
            |--------|-------|
            | Composite Score | ${report.composite_score.toFixed(3)} |
            | Baseline | ${report.baseline_score.toFixed(3)} |
            | Delta | ${report.delta.toFixed(3)} |
            | Drift Detected | ${report.drift_detected ? '🔴 YES' : '🟢 NO'} |
            `;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body
            });

      - name: Fail if score dropped >5%
        run: |
          python -c "
          import json, sys
          r = json.load(open('eval_report.json'))
          if r['delta'] < -0.05:
              print(f'❌ Composite score dropped {r[\"delta\"]*100:.1f}% — blocking merge')
              sys.exit(1)
          print(f'✅ Score delta: {r[\"delta\"]*100:.1f}% — within threshold')
          "
```

#### 3.3 Prometheus Metrics Endpoint

```python
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

quality_score_gauge  = Gauge("driftscope_quality_score",   "Latest composite quality score", ["model_version"])
drift_detected_gauge = Gauge("driftscope_drift_detected",  "1 if drift detected, 0 otherwise", ["model_version"])
judge_pass_gauge     = Gauge("driftscope_judge_pass_rate", "LLM judge pass rate", ["model_version"])

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

#### 3.4 Docker Compose Stack

One command runs the entire platform locally:

```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql://drift:drift@db:5432/driftscope
    depends_on: [db]

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: driftscope
      POSTGRES_USER: drift
      POSTGRES_PASSWORD: drift
    volumes:
      - postgres-data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres-data:
  grafana-data:
```

```bash
docker compose up --build
```

**Implemented endpoints and commands:**
```
GET /drift/{model_version}  → Mann-Whitney drift status
GET /metrics                → Prometheus text metrics

docker compose up --build   → API + Postgres + Prometheus + Grafana
```

**Deliverable checklist:**
- [x] `detect_drift()` returns correct `p_value`, `effect_size`, `drift_detected` fields
- [x] `GET /drift/{model_version}` endpoint live and documented
- [x] `.github/workflows/eval.yml` triggers on PRs
- [x] PR comment posted with quality score table
- [x] Merge blocked when composite drops >5%
- [x] `/metrics` endpoint returns valid Prometheus text format
- [x] Grafana dashboard imported with all 3 gauges
- [x] `docker compose up` starts all 4 services cleanly

---

### Phase 4 — Provider-Change Canary & Empirical Study

**Goal:** Build the second novel feature — a nightly canary that uses SBERT centroid tracking to detect when a provider silently swaps their underlying model. Run against Gemini and a local Ollama model to observe real drift over time.

#### 4.1 Embedding Centroid Tracking

This borrows the technique from Zanbaghi et al. (arXiv:2511.15992), which uses Sentence-BERT centroid tracking to detect backdoored LLMs with **92.5% accuracy**, applied here to provider-update detection.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer('all-MiniLM-L6-v2')

def compute_centroid(responses: list[str]) -> np.ndarray:
    """Mean embedding of a set of responses — the 'centre of mass' of the model's output space."""
    embeddings = sbert.encode(responses, normalize_embeddings=True)
    return embeddings.mean(axis=0)

def centroid_drift(current: np.ndarray, previous: np.ndarray) -> float:
    """Cosine distance between two centroids. 0 = identical, 1 = orthogonal."""
    return 1.0 - float(np.dot(current, previous) /
                       (np.linalg.norm(current) * np.linalg.norm(previous) + 1e-9))
```

#### 4.2 Nightly Canary Job

```python
async def run_canary(provider: str, golden_case_ids: list[str], db_session) -> dict:
    """
    Run a fixed golden set against the live provider API.
    Compute today's centroid and compare to the 7-day rolling centroid.
    Send an email alert if drift > 0.05.
    """
    responses = [await llm_client.complete(get_prompt(cid)) for cid in golden_case_ids]
    current_centroid = compute_centroid(responses)

    prev_centroid = get_latest_centroid(db_session, provider)  # from centroid_history table
    drift = centroid_drift(current_centroid, prev_centroid) if prev_centroid is not None else 0.0

    store_centroid(db_session, provider, current_centroid, drift)

    if drift > 0.05:
        await send_email_alert(provider, drift)

    return {"provider": provider, "drift_score": round(drift, 4), "alert_sent": drift > 0.05}
```

**Schema addition:**
```sql
CREATE TABLE centroid_history (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider     VARCHAR(50),
    centroid     FLOAT[],
    drift_score  FLOAT,
    recorded_at  TIMESTAMPTZ DEFAULT now()
);
```

#### 4.3 GitHub Actions Cron Schedule

Two workflows run DriftScope on a schedule. Both also accept `workflow_dispatch` for manual triggering.

| Workflow | File | Cron (UTC) | IST | Purpose |
|---|---|---|---|---|
| Nightly Canary | `canary.yml` | `53 4 * * *` | 10:23 | Centroid snapshot + drift check |
| Scheduled Eval | `monitor.yml` | `43 5,17 * * *` | 11:13 / 23:13 | Golden suite, twice daily |

**Schedules are UTC.** GitHub Actions cron has no timezone setting, while the Actions tab renders start times in your local timezone — the two never read the same.

**Start times are approximate.** GitHub queues `schedule` events behind on-demand jobs and commonly starts them one to three hours late. Both crons sit mid-hour and away from `:00` and midnight UTC, the most contended slots, which reduces the lag but can't remove it. Nothing downstream depends on the exact minute — `/metrics` averages over a rolling 24h window, so a late run still lands inside it. Where exact timing matters, drive the workflow from an external scheduler via `workflow_dispatch`.

The canary runs against `gemini` only. Ollama needs a local `ollama serve` and a pulled model, so it can't run on a hosted runner — add it explicitly (`--providers gemini ollama`) when running locally.

#### 4.4 Email Integration

DriftScope sends two kinds of email, both through `app/core/notify.py` using Python's built-in `smtplib` — no extra dependencies.

| Kind | Sent by | When |
|---|---|---|
| **Drift alert** | `app/core/canary.py` | Canary centroid drift crosses the threshold |
| **Run summary** | `scripts/notify_run.py` | End of every scheduled run, pass or fail |

Run summaries are what tell you the monitoring is alive. GitHub only emails on workflow *failure*, and the drift alert only fires above the threshold, so neither one distinguishes a healthy system from a job that stopped running. Each summary carries the run's status, a link back to the Actions run, and the case counts and scores the job recorded:

```
DriftScope scheduled job: monitor
========================================
Status     : OK
Finished   : 2026-08-06T10:31:44+00:00
Run        : https://github.com/Halcyonic-01/DriftScope/actions/runs/…

Result
----------------------------------------
  cases_failed          : 0
  cases_succeeded       : 20
  mean_composite_score  : 0.7306
  provider              : gemini
```

Both workflows invoke the notifier with `if: always()`, so failed runs are reported too. The step never fails a build: if SMTP is unconfigured it logs a warning and skips, and delivery errors are caught and logged. Monitoring results matter more than the notification about them.

**Configuration** — three repository secrets under *Settings → Secrets and variables → Actions*:

| Secret | Value |
|---|---|
| `ALERT_EMAIL` | where alerts and summaries go |
| `SMTP_USER` | the sending account |
| `SMTP_PASSWORD` | an **app password**, not your login password ([Gmail](https://support.google.com/accounts/answer/185833)) |

`SMTP_HOST` and `SMTP_PORT` default to `smtp.gmail.com:465`. Locally, the same values come from `.env`. Works with Gmail, Outlook, or any SMTP provider.

#### 4.5 Multi-Provider Comparison Dashboard

A Grafana panel is added showing side-by-side quality scores across providers:

```
driftscope_quality_score{model_version="gemini-3.5-flash-lite"}
driftscope_quality_score{model_version="llama3-local"}
```

#### 4.6 30-Day Empirical Study

The canary runs daily for 30 days, logging:
- Date
- Provider (`gemini`, `ollama`)
- Centroid drift score
- Alert threshold crossed? (yes/no)
- Any corroborating external evidence (changelog entries, community reports)

**This data does not exist publicly — it is a novel contribution to the space.**

**Deliverable checklist:**
- [ ] `run_canary()` computes centroid and drift score correctly
- [ ] `centroid_history` table storing daily snapshots
- [ ] Nightly cron job running at `0 2 * * *`
- [ ] Email alert fires when `drift_score > 0.05`
- [ ] Multi-provider comparison panel in Grafana
- [ ] 30 days of daily canary data logged

---

## 📚 Research Base

| # | Paper | Key Insight |
|---|-------|------------|
| 1 | [Murphy & Underwood, ACM Queue 2025](https://queue.acm.org/detail.cfm?id=3762989) | Provider-version problem; production model quality = #1 unsolved MLOps problem. Motivates the canary feature. |
| 2 | [arXiv:2602.11165 (2026)](https://arxiv.org/pdf/2602.11165) | Models achieve >99% cosine similarity despite <8% BLEU overlap → validates embedding-based detection |
| 3 | [Paunova DTE (2025)](https://github.com/epaunova/dte) | Open-source stat drift for RAG using significance tests. Directly inspired the Mann-Whitney detector. |
| 4 | [Zanbaghi et al., arXiv:2511.15992 (2025)](https://arxiv.org/abs/2511.15992) | SBERT centroid tracking → 92.5% accuracy detecting backdoored LLMs. Validates Phase 4 canary approach. |
| 5 | [Gu et al., arXiv:2411.15594 (2024)](https://arxiv.org/abs/2411.15594) | LLM-as-judge survey: biases (position, verbosity), mitigations, 80-90% human agreement. |
| 6 | [arXiv:2501.18243 (2025)](https://arxiv.org/pdf/2501.18243) | Statistical multi-metric evaluation: Mann-Whitney U + effect size theory for LLM system comparisons. |

---

## ✅ Quick-Start Checklist

<details>
<summary><strong>Phase 1 — Foundation</strong></summary>

- [ ] `pip install sentence-transformers fastapi uvicorn psycopg2-binary alembic google-generativeai`
- [ ] Create PostgreSQL schema: `golden_cases`, `eval_results` + Alembic migration
- [ ] Implement `embed()` + `cosine_sim()` utilities with pytest unit tests
- [ ] Build `POST /cases` and `POST /cases/{id}/run` FastAPI endpoints
- [ ] Wire Gemini Flash Lite + Ollama behind unified `LLMClient` (factory pattern)
- [ ] Seed 20+ golden test cases for the chosen domain

</details>

<details>
<summary><strong>Phase 2 — Intelligence Layer</strong></summary>

- [ ] Add LLM-as-judge with structured rubric prompt (Gemini JSON mode)
- [ ] Implement cost guard: skip judge if cosine >= 0.65 and not safety-tagged
- [ ] Implement composite score formula with configurable weights
- [ ] Build `GET /reports/{model_version}` aggregate endpoint
- [ ] Write integration tests with mocked LLM responses (80%+ coverage)
- [ ] Add judge reason strings to `eval_results` for debugging

</details>

<details>
<summary><strong>Phase 3 — Drift Detection & DevOps</strong></summary>

- [ ] Implement `detect_drift()` using `scipy.stats.mannwhitneyu`
- [ ] Add rolling window queries to `eval_results` (24h vs 7-day baseline)
- [ ] Build `GET /drift/{model_version}` endpoint returning p-value + effect size
- [ ] Create `.github/workflows/eval.yml` GitHub Actions workflow
- [ ] Implement PR comment posting with quality score + drift status
- [ ] Add merge block on composite score drop > 5% from baseline
- [ ] Export `/metrics` Prometheus endpoint + import Grafana dashboard JSON
- [ ] Package as Docker Compose: FastAPI + PostgreSQL + Grafana + Prometheus

</details>

<details>
<summary><strong>Phase 4 — Provider-Change Canary</strong></summary>

- [ ] Implement `run_canary()` with embedding centroid tracking
- [ ] Add `centroid_history` table to PostgreSQL schema
- [ ] Schedule nightly canary via GitHub Actions cron (`0 2 * * *`)
- [ ] Set up email alert via `smtplib` for centroid drift > 0.05
- [ ] Add multi-provider comparison panel to Grafana dashboard
- [ ] Run canary for 30 days — log all centroid drift values

</details>

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Halcyonic-01/DriftScope.git
cd DriftScope

# 2. Copy env file and fill in your API key
cp .env.example .env
# Add your GEMINI_API_KEY from https://aistudio.google.com

# 3. Spin up the full stack (FastAPI + PostgreSQL + Prometheus + Grafana)
docker compose up --build

# 4. Run database migrations
docker compose exec api alembic upgrade head

# 5. Seed golden test cases
docker compose exec api python scripts/seed.py

# 6. Open the API docs
open http://localhost:8000/docs

# 7. Open Grafana dashboard
open http://localhost:3000
```

The `api` service builds from local source, so **rebuild it after any code or migration change** — plain `docker compose up -d` reuses the existing image:

```bash
docker compose up -d --build api
```

### 📊 Where the dashboard reads from

Grafana renders what Prometheus scrapes from the API's `/metrics`, which the API derives from the database. So the dashboard shows whichever database the **`api` container** is pointed at:

| Variable | Read by | Typical value |
|---|---|---|
| `DRIFTSCOPE_DB_URL` | the `api` container (via compose) | hosted DB, e.g. the Neon instance the workflows write to |
| `DATABASE_URL` | host-side tooling — alembic, `seed.py`, `run_monitor.py` | `localhost` |

They're deliberately separate: compose auto-loads `.env`, and a `localhost` URL inside a container resolves to the container itself. Set `DRIFTSCOPE_DB_URL` to the same database your scheduled workflows write to and the dashboard reflects those runs.

If panels read **No data**, check the chain end to end — `docker compose ps` (is `api` up?), then http://localhost:9090/targets (is the scrape target `up`?), then `curl localhost:8000/metrics`.

> The **LLM Calls / sec** panel stays empty unless evals run through the API process itself. Scheduled runs execute on GitHub runners, which never touch this process's in-memory counters — the `driftscope_job_*` metrics cover those instead.

### 🔑 API Auth (optional)

By default the API is open — no key required. To lock it down, set `API_KEY` in `.env` to any random string, then send it back on every request via the `X-API-Key` header:

```bash
curl http://localhost:8000/cases -H "X-API-Key: your-secret-here"
```

`/health` and `/metrics` are always open (liveness probes and Prometheus scraping don't send custom headers). Everything else — `/cases`, `/drift`, `/reports` — requires the key once `API_KEY` is set.

---

## 📡 Keeping Monitoring Alive

DriftScope only shows data if evals actually run. The `/metrics` gauges average over a rolling **24-hour** window, and drift detection needs **≥10 results in 24h plus ≥30 in the prior 7 days**. If nothing runs on a schedule, the dashboard goes blank and drift detection reports `insufficient_data` forever.

`.github/workflows/monitor.yml` runs the golden suite twice daily — `43 5,17 * * *` **UTC** (11:13 / 23:13 IST) — to keep the window populated. Against ~21 golden cases that's ~42 results/day, clearing both thresholds with margin. Run it by hand with:

```bash
python scripts/run_monitor.py --provider gemini --model-version gemini-3.5-flash-lite
```

Cron is UTC and GitHub starts scheduled runs up to a few hours late — see [§4.3](#43-github-actions-cron-schedule).

### Knowing the jobs ran

Every scheduled run emails a summary (`scripts/notify_run.py`) whether it passed or failed, so a stalled job is visible from your inbox rather than only from the dashboard. See [§4.4](#44-email-integration).

### Metrics that guard the monitoring itself

Quality gauges alone can't tell "the model is fine" from "the eval job died". These close that gap:

| Metric | Answers |
|---|---|
| `driftscope_eval_last_run_timestamp_seconds` | When did evals last run? (stale = job is dead) |
| `driftscope_canary_last_run_timestamp_seconds` | When did the canary last run? |
| `driftscope_eval_runs_total` | How many results landed in the last 24h? |
| `driftscope_drift_insufficient_data` | Is `drift_detected=0` real, or are we blind? |
| `driftscope_llm_requests_total{outcome}` | Is the provider erroring or quota-blocked? |
| `driftscope_http_requests_total{status}` | Is the API itself healthy? |

> `driftscope_drift_detected=0` on its own is ambiguous — it means both "no drift" **and** "not enough data to tell". Always read it alongside `driftscope_drift_insufficient_data`.

### Alerting

`alerts.yml` ships 8 Prometheus rules covering quality drift, canary centroid shift, stalled eval/canary jobs, blind drift detection, and LLM/API error rates. Firing alerts appear at **http://localhost:9090/alerts**.

Prometheus only *evaluates* alerts — to route them to email or Slack you additionally need an [Alertmanager](https://prometheus.io/docs/alerting/alertmanager/) service. The two SMTP paths in `app/core/notify.py` (the canary drift alert and the per-run summaries) are independent of Prometheus and already work.

---

## 🧪 Running Tests

Tests live in three tiers under `tests/`:

| Tier | Location | What it covers | Needs a DB? |
|------|----------|-----------------|-------------|
| Unit | `tests/test_*.py` | Pure logic — scoring, drift stats, embeddings, LLM factory — with mocked dependencies | No |
| Integration | `tests/integration/` | Individual FastAPI routes against a real Postgres session (via the app's own `mock` LLM provider) | Yes |
| End-to-end | `tests/e2e/` | Full multi-step workflows through the HTTP API — create case → run evals → report → drift → metrics; canary run → `/metrics` | Yes |

```bash
# Everything
pytest

# Just unit tests (fast, no DB)
pytest -m "not integration and not e2e"

# Just integration or e2e
pytest -m integration
pytest -m e2e
```

Integration/e2e tests run against `DATABASE_URL` (from `.env`) by default. Point `TEST_DATABASE_URL` at a separate database to isolate test runs from dev data:

```bash
TEST_DATABASE_URL=postgresql://drift:drift@localhost:5432/driftscope_test pytest -m "integration or e2e"
```

Every test runs inside a transaction that's rolled back afterwards (including nested `commit()` calls made by route handlers), so nothing written during a test run is ever persisted — safe to point at a shared dev database. If no database is reachable, integration/e2e tests are skipped automatically rather than failing.

---

<div align="center">
  <sub>Built as a learning project · June 2026 · Research: ACM Queue 2025 · arXiv:2602.11165 · arXiv:2511.15992 · Paunova DTE 2025 · arXiv:2411.15594 · arXiv:2501.18243</sub>
</div>
