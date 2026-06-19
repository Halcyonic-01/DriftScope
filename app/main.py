"""
app/main.py

FastAPI application entry point.
Start the server with:  uvicorn app.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(
    title="DriftScope",
    description="LLM Quality Monitoring Platform",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check():
    """Quick liveness check — returns OK if the server is running."""
    return {"status": "ok", "service": "driftscope"}
