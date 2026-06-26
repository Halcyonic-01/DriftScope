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
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)

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
│  ┌──────────────┐    ┌──────────────────┐                          │
│  │  Prometheus  │───▶│    Grafana       │                          │
│  │  /metrics    │    │    Dashboard     │                          │
│  └──────────────┘    └──────────────────┘                          │
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
    recorded_at     TIMESTAMPTZ DEFAULT now()
);
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| API Framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + Alembic (migrations) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| ML / Math | PyTorch, FAISS, scikit-learn, scipy |
| Statistics | `scipy.stats.mannwhitneyu` |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |
| Observability | Prometheus + Grafana |
| Infrastructure | Docker Compose, Redis |
| LLM (Judge + Canary) | **Google Gemini 2.5 Flash** (free tier) |

---

## 🤖 Free LLM Choice

**Google Gemini 2.5 Flash** is used as the LLM for both the judge and the canary runs. Here’s why it’s the best free option for this project:

| Criterion | Gemini 2.5 Flash |
|-----------|-----------------|
| **Cost** | Free tier via [Google AI Studio](https://aistudio.google.com) — 15 RPM, 1M tokens/day |
| **Structured output** | Native JSON mode — critical for the LLM-as-judge `{pass, reason}` schema |
| **Speed** | Sub-second latency — fast enough for nightly canary runs |
| **Context window** | 1M tokens — handles long responses without truncation |
| **Python SDK** | `google-generativeai` — simple, well-documented |
| **Quality** | Matches or beats GPT-3.5 on evaluation tasks at zero cost |

```bash
pip install google-generativeai
```

```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

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

**Gemini 2.5 Flash** is wired as the primary provider, with **Ollama** (local) as the fallback. The factory pattern means providers can be swapped without touching any business logic.

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

#### 2.1 LLM-as-Judge (Gemini 2.5 Flash)

A structured rubric prompt is implemented that returns `{pass: bool, reason: str}`:

```python
import google.generativeai as genai
import json

model = genai.GenerativeModel("gemini-2.5-flash")

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

A `.github/workflows/canary.yml` workflow is created:

```yaml
name: DriftScope Nightly Canary

on:
  schedule:
    - cron: "0 2 * * *"   # 2 AM UTC every night
  workflow_dispatch:        # also allow manual trigger

jobs:
  canary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run canary
        run: python scripts/run_canary.py --providers gemini ollama
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          ALERT_EMAIL: ${{ secrets.ALERT_EMAIL }}
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
```

#### 4.4 Email Alert Integration

Python's built-in `smtplib` is used — no extra dependencies needed:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

async def send_email_alert(
    provider: str,
    drift_score: float,
    to_email: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str
):
    subject = f"🚨 DriftScope Canary Alert — {provider} drift detected"

    body = f"""
    DriftScope Canary Alert
    ========================
    Provider      : {provider}
    Centroid Drift: {drift_score:.4f}  (threshold: 0.05)
    Detected at   : {datetime.utcnow().isoformat()}Z

    A silent model update may have occurred.
    Review your eval results at http://localhost:3000 (Grafana).
    """

    msg = MIMEMultipart()
    msg["From"]    = smtp_user
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_email, msg.as_string())
```

Configure via environment variables — works with Gmail, Outlook, or any SMTP provider.

#### 4.5 Multi-Provider Comparison Dashboard

A Grafana panel is added showing side-by-side quality scores across providers:

```
driftscope_quality_score{model_version="gemini-2.5-flash"}
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
- [ ] Wire Gemini 2.5 Flash + Ollama behind unified `LLMClient` (factory pattern)
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

---

<div align="center">
  <sub>Built as a learning project · June 2026 · Research: ACM Queue 2025 · arXiv:2602.11165 · arXiv:2511.15992 · Paunova DTE 2025 · arXiv:2411.15594 · arXiv:2501.18243</sub>
</div>
