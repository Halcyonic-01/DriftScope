<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/DriftScope-LLM%20Quality%20Monitor-6366f1?style=for-the-badge&labelColor=0f0f0f">
  <img alt="DriftScope" src="https://img.shields.io/badge/DriftScope-LLM%20Quality%20Monitor-6366f1?style=for-the-badge">
</picture>

# 🔭 DriftScope

> **A full-stack LLM quality monitoring platform built from scratch.**  
> Detect silent model regressions, run statistically-grounded drift detection, and catch provider-side model swaps before your users do.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📖 Table of Contents

- [Why DriftScope](#-why-driftscope)
- [What You're Building](#-what-youre-building)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Phase Implementation Plan](#-phase-implementation-plan)
  - [Phase 1 — Foundation (Weeks 1–2)](#phase-1--foundation-weeks-12)
  - [Phase 2 — Intelligence Layer (Weeks 3–4)](#phase-2--intelligence-layer-weeks-34)
  - [Phase 3 — Drift Detection & DevOps (Weeks 5–6)](#phase-3--drift-detection--devops-weeks-56)
  - [Phase 4 — Provider-Change Canary & Empirical Study (Weeks 7–8)](#phase-4--provider-change-canary--empirical-study-weeks-78)
- [Competitive Landscape](#-competitive-landscape)
- [Research Base](#-research-base)
- [Quick-Start Checklist](#-quick-start-checklist)
- [Getting Started](#-getting-started)

---

## 🚨 Why DriftScope

The underlying pain is real even if the tooling space is crowded. Enterprises spend approximately **$14,200 per employee annually** dealing with LLM hallucinations and silent quality regressions — roughly 4.3 hours per worker per week on fact-checking and error correction. Teams that ship without continuous evaluation typically discover regressions from **customer complaints days after the fact**.

> *"There is no guarantee that the system named GPT-4o at 16:18 will be the same system at 18:16."*  
> — Murphy & Underwood, ACM Queue 2025

The specific gap DriftScope addresses: **no existing open-source tool detects when a provider like OpenAI or Anthropic silently updates their model** — your application's behaviour changes with no API version bump, no changelog, no warning.

---

## 🏗️ What You're Building

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
│  │    history   │    │  · OpenAI        │    │  CI/CD Gate       │ │
│  └──────────────┘    │  · Anthropic     │    │  · PR Comments    │ │
│                      │  · Ollama (local)│    │  · Merge Blocking │ │
│                      └──────────────────┘    └───────────────────┘ │
│                                                                     │
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
    version_tag    VARCHAR(50),   -- e.g. "v1.2-gpt4o"
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
    provider        VARCHAR(50),  -- "openai", "anthropic", "local"
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
| Local LLM | Ollama |

---

## 📅 Phase Implementation Plan

---

### Phase 1 — Foundation (Weeks 1–2)

**Goal:** Stand up the core data layer, embedding utilities, and REST API skeleton. By the end of week 2 you should be able to store golden test cases and run a basic cosine-similarity eval against a live model.

#### 1.1 Environment Setup

```bash
pip install sentence-transformers fastapi uvicorn psycopg2-binary alembic \
            scipy numpy pytest httpx python-dotenv openai anthropic
```

Create a `.env.example`:
```
DATABASE_URL=postgresql://user:pass@localhost:5432/driftscope
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

#### 1.2 PostgreSQL Schema + Alembic Migration

- Install Alembic and initialise: `alembic init alembic/`
- Create the `golden_cases` and `eval_results` tables (see schema above)
- Run initial migration: `alembic upgrade head`
- Write a seed script that inserts 20+ golden test cases for your chosen domain (e.g. medical Q&A, legal summarisation, customer support)

**Key design decision:** Store `expected_topics` (what the model should cover) and `safety_rules` (natural-language rules for the LLM judge), **not** verbatim expected strings. This makes evals robust to non-determinism.

#### 1.3 Embedding Utilities

Implement two core utility functions with full pytest unit tests:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # free, runs locally, fast

def embed(text: str) -> np.ndarray:
    """Return a unit-normalised embedding vector."""
    return model.encode(text, normalize_embeddings=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised vectors."""
    return float(np.dot(a, b))
```

> **Why `all-MiniLM-L6-v2`?**  
> 384-dimensional, 22M params, runs in ~5ms on CPU. arXiv:2602.11165 shows models achieve >99% cosine similarity with gold references despite <8% BLEU-1 overlap — embeddings capture *meaning*, lexical metrics don't.

**Tests to write:**
- `test_embed_returns_unit_vector()`
- `test_cosine_sim_identical_texts()`
- `test_cosine_sim_orthogonal_texts()`
- `test_cosine_sim_threshold_at_0_80()`

#### 1.4 Unified LLM Client (Factory Pattern)

```python
class LLMClient:
    def __init__(self, provider: str):  # "openai" | "anthropic" | "local"
        ...
    def complete(self, prompt: str, schema: dict | None = None) -> dict:
        ...

def get_client(provider: str) -> LLMClient:
    """Factory — swap providers without touching business logic."""
    ...
```

- Wire OpenAI (`gpt-4o`) and Anthropic (`claude-3-5-sonnet`) behind the same interface
- Support optional structured output via `schema` param (used by LLM-as-judge in Phase 2)

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
- [ ] `POST /cases/{id}/run` calls the LLM, embeds the response, stores cosine score
- [ ] Pytest suite passes for embedding utilities

---

### Phase 2 — Intelligence Layer (Weeks 3–4)

**Goal:** Add the LLM-as-judge, cost guard, composite scoring, and aggregated reporting. By end of week 4 you should have an 80%+ covered integration test suite with mocked LLM responses.

#### 2.1 LLM-as-Judge

Implement a structured rubric prompt that returns `{pass: bool, reason: str}`:

```python
def judge_response(response: str, rule: str, client: LLMClient) -> dict:
    prompt = f"""You are a strict evaluator. Answer in JSON only.

Rule: {rule}
Response to evaluate: {response}

Does the response satisfy the rule?
Respond with: {{"pass": true/false, "reason": "one sentence explanation"}}"""

    return client.complete(prompt, schema={"pass": bool, "reason": str})
```

**Known biases to mitigate (from arXiv:2411.15594):**
- **Position bias:** If comparing multiple responses, randomise their order
- **Verbosity bias:** Longer responses score higher regardless of quality — keep rubrics length-agnostic
- **Self-enhancement bias:** Don't use the same model as judge and evaluee

#### 2.2 Cost Guard

The LLM judge is expensive. Only invoke it when:
1. `cosine_score < 0.85` (borderline semantic match), **OR**
2. The case has `"safety"` in its tags

```python
def should_invoke_judge(cosine_score: float, case_tags: list[str]) -> bool:
    return cosine_score < 0.85 or "safety" in case_tags
```

This typically limits judge invocations to **<15% of evals**, keeping costs manageable at scale.

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

Store `cosine_score`, `judge_score`, and `composite_score` separately in `eval_results` for full auditability.

#### 2.4 Reporting Endpoint

```
GET /reports/{model_version}
```

Returns aggregated stats for a model version:
```json
{
  "model_version": "v1.2-gpt4o",
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

Use `pytest` + `httpx.AsyncClient` + `unittest.mock` to mock LLM responses:

```python
@pytest.mark.asyncio
async def test_run_eval_stores_composite_score(mock_llm_client, async_client, db_session):
    # Arrange: seed one golden case
    case = await create_test_case(db_session, domain="medical")
    mock_llm_client.complete.return_value = "Patient should rest and hydrate."

    # Act
    resp = await async_client.post(f"/cases/{case.case_id}/run",
                                   json={"provider": "openai", "model_version": "v1.2-gpt4o"})

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

### Phase 3 — Drift Detection & DevOps (Weeks 5–6)

**Goal:** Add statistically-grounded drift detection, wire it into a GitHub Actions CI gate, and launch the full observability stack. By end of week 6, a Docker Compose `up` spins the entire platform locally.

#### 3.1 Mann-Whitney Drift Detector

The **novel core** of DriftScope. Rather than comparing single scores, compare *distributions*.

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
Cosine score distributions are rarely Gaussian. Mann-Whitney U is non-parametric — no normality assumption. Cohen's d adds *practical* significance on top of *statistical* significance, preventing false alerts on tiny real-world differences.

**API endpoint:**
```
GET /drift/{model_version}
```
Response:
```json
{
  "model_version": "v1.2-gpt4o",
  "drift_detected": true,
  "p_value": 0.0231,
  "effect_size": 0.412,
  "today_mean": 0.801,
  "baseline_mean": 0.873
}
```

#### 3.2 GitHub Actions CI/CD Gate

Create `.github/workflows/eval.yml`:

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
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

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

`docker-compose.yml` — single command to run the entire platform:

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

Anyone should be able to run the full stack with:
```bash
docker compose up --build
```

**Deliverable checklist:**
- [ ] `detect_drift()` returns correct `p_value`, `effect_size`, `drift_detected` fields
- [ ] `GET /drift/{model_version}` endpoint live and documented
- [ ] `.github/workflows/eval.yml` triggers on PRs
- [ ] PR comment posted with quality score table
- [ ] Merge blocked when composite drops >5%
- [ ] `/metrics` endpoint returns valid Prometheus text format
- [ ] Grafana dashboard imported with all 3 gauges
- [ ] `docker compose up` starts all 4 services cleanly

---

### Phase 4 — Provider-Change Canary & Empirical Study (Weeks 7–8)

**Goal:** Build the second novel feature — a nightly canary that uses SBERT centroid tracking to detect when OpenAI or Anthropic silently swaps their underlying model. Run it for 30 days and publish your findings.

#### 4.1 Embedding Centroid Tracking

The technique is taken from Zanbaghi et al. (arXiv:2511.15992), which uses Sentence-BERT centroid tracking to detect backdoored LLMs with **92.5% accuracy**. DriftScope applies it to provider-update detection.

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
    Alert if drift > 0.05.
    """
    responses = [await llm_client.complete(get_prompt(cid)) for cid in golden_case_ids]
    current_centroid = compute_centroid(responses)

    prev_centroid = get_latest_centroid(db_session, provider)  # from centroid_history table
    drift = centroid_drift(current_centroid, prev_centroid) if prev_centroid is not None else 0.0

    store_centroid(db_session, provider, current_centroid, drift)

    if drift > 0.05:
        await send_alert(provider, drift)   # Slack webhook or email

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

Add to `.github/workflows/canary.yml`:

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
        run: python scripts/run_canary.py --providers openai anthropic
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

#### 4.4 Slack Alert Integration

```python
import httpx

async def send_slack_alert(provider: str, drift_score: float, webhook_url: str):
    payload = {
        "text": f"🚨 *DriftScope Canary Alert*",
        "blocks": [{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Provider:* `{provider}`\n"
                    f"*Centroid Drift:* `{drift_score:.4f}` (threshold: 0.05)\n"
                    f"*Detected at:* {datetime.utcnow().isoformat()}Z\n\n"
                    f"_A silent model update may have occurred. Review your eval results._"
                )
            }
        }]
    }
    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json=payload)
```

#### 4.5 Multi-Provider Comparison Dashboard

Add a Grafana panel showing side-by-side quality scores across providers:

```
driftscope_quality_score{model_version="gpt-4o"}
driftscope_quality_score{model_version="claude-3-5-sonnet"}
driftscope_quality_score{model_version="llama3-local"}
```

#### 4.6 30-Day Empirical Study

Run the canary daily for 30 days. Log:
- Date
- Provider (`openai`, `anthropic`, `local`)
- Centroid drift score
- Alert threshold crossed? (yes/no)
- Any corroborating external evidence (changelog entries, community reports)

**This data does not exist publicly — it is your novel contribution.**

#### 4.7 Write-Up

Publish a blog post or short research note:
> *"What we observed running a 30-day canary against OpenAI and Anthropic"*

Even if you detect zero changes, the null result is interesting and publishable. Target: GitHub README + dev.to or arXiv.

**Deliverable checklist:**
- [ ] `run_canary()` computes centroid and drift score correctly
- [ ] `centroid_history` table storing daily snapshots
- [ ] Nightly cron job running at `0 2 * * *`
- [ ] Slack/email alert fires when `drift_score > 0.05`
- [ ] Multi-provider comparison panel in Grafana
- [ ] 30 days of daily canary data logged
- [ ] Write-up published (blog post / arXiv note)

---

## 🗺️ Competitive Landscape

| Tool | What it actually does | What DriftScope adds for learning |
|------|----------------------|----------------------------------|
| **Arize Phoenix** (9,100+ ⭐) | Tracing + LLM-judge + embedding drift + composite weighted score + CI evals | You build the same thing from scratch. You learn *why* each design decision exists. Novel additions: stat drift + canary. |
| **Evidently AI** | Input drift + LLM output eval + CI/CD GitHub Action that blocks PRs. 100+ metrics. Apache 2.0. | Evidently's CI gate requires their config DSL. Yours is pure Python + GitHub Actions YAML. Full control, full understanding. |
| **DeepEval / Confident AI** | 50+ research-backed metrics. LLM-as-judge. CI integration. 80-90% human agreement. Used by OpenAI, Google, Microsoft. | DeepEval abstracts the scorer away. You implement it yourself. You understand what '80-90% agreement' means. |
| **Langfuse** (21,000+ ⭐) | Tracing + prompt management + eval. MIT licensed. Acquired by ClickHouse 2025. | Langfuse requires building your own eval layer. DriftScope builds that layer explicitly. Stat drift + canary not in Langfuse. |
| **(gap)** | Rolling Mann-Whitney U test on output embedding distributions across time windows. No packaged tool ships this. | ✅ DriftScope ships this. Genuinely novel. Math-based drift gating, not threshold vibes. |
| **(gap)** | Nightly provider-change canary detecting silent OpenAI/Anthropic model updates. No existing tool ships this. | ✅ DriftScope ships this. Genuinely novel. 30-day empirical data is publishable. |

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
<summary><strong>Week 1–2</strong></summary>

- [ ] `pip install sentence-transformers fastapi uvicorn psycopg2-binary alembic`
- [ ] Create PostgreSQL schema: `golden_cases`, `eval_results` + Alembic migration
- [ ] Implement `embed()` + `cosine_sim()` utilities with pytest unit tests
- [ ] Build `POST /cases` and `POST /cases/{id}/run` FastAPI endpoints
- [ ] Wire OpenAI + Anthropic behind unified `LLMClient` (factory pattern)
- [ ] Seed 20+ golden test cases for your chosen domain

</details>

<details>
<summary><strong>Week 3–4</strong></summary>

- [ ] Add LLM-as-judge with structured rubric prompt template
- [ ] Implement cost guard: skip judge if cosine >= 0.85 and not safety-tagged
- [ ] Implement composite score formula with configurable weights
- [ ] Build `GET /reports/{model_version}` aggregate endpoint
- [ ] Write integration tests with mocked LLM responses (80%+ coverage)
- [ ] Add judge reason strings to `eval_results` for debugging

</details>

<details>
<summary><strong>Week 5–6</strong></summary>

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
<summary><strong>Week 7–8</strong></summary>

- [ ] Implement `run_canary()` with embedding centroid tracking
- [ ] Add `centroid_history` table to PostgreSQL schema
- [ ] Schedule nightly canary via GitHub Actions cron (`0 2 * * *`)
- [ ] Set up Slack/email alert webhook for centroid drift > 0.05
- [ ] Add multi-provider comparison panel to Grafana dashboard
- [ ] Run canary for 30 days — log all centroid drift values
- [ ] Write up findings: blog post or short arXiv note on provider drift observations
- [ ] Publish to GitHub with full README, architecture diagram, and setup guide

</details>

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Halcyonic-01/DriftScope.git
cd DriftScope

# 2. Copy env file and fill in your API keys
cp .env.example .env

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

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built as a learning project · June 2026 · All competitive claims verified against June 2026 tool documentation</sub>
</div>
