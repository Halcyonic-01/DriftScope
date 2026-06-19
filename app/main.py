"""
app/main.py

FastAPI application entry point.

LIFESPAN:
  Modern FastAPI uses a "lifespan" context manager for startup/shutdown logic.
  We use it to verify the DB connection is alive when the server starts.
  This prevents silent failures where the app starts but can't talk to the DB.

Start the server with:
  source .venv/bin/activate
  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE at startup (before the first request) and once on shutdown.
    
    The code BEFORE 'yield' = startup logic.
    The code AFTER  'yield' = shutdown logic.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("DriftScope starting up…")

    # Verify the database is reachable
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection verified.")
    except Exception as exc:
        logger.error("❌ Cannot connect to database: %s", exc)
        raise

    logger.info("✅ DriftScope ready — environment: %s", settings.app_env)

    yield  # ← server runs here, handling requests

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("DriftScope shutting down…")
    engine.dispose()   # close all DB connections cleanly


# Create the FastAPI app
app = FastAPI(
    title="DriftScope",
    description=(
        "LLM Quality Monitoring Platform — "
        "detect semantic drift in your AI responses over time."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── Register Routers ──────────────────────────────────────────────────────────
# Import here (not at top) to avoid circular imports at collection time
from app.api.routes.cases import router as cases_router  # noqa: E402

app.include_router(cases_router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """Quick liveness check — returns OK if the server is running."""
    return {
        "status": "ok",
        "service": "driftscope",
        "environment": settings.app_env,
    }
