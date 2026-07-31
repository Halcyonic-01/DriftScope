"""
app/core/instrumentation.py

Process-level Prometheus metrics for operational health.

These are distinct from the quality metrics in app/api/routes/metrics.py:
those are derived from eval_results in the database and describe MODEL
health; these are counters and histograms held in memory that describe
SERVICE health -- request rates, latencies, and LLM provider failures.

Without these, an outage is invisible on the dashboard. If Gemini starts
returning 429s, every eval fails but the quality gauges just stop
updating, which looks identical to "nothing scheduled right now".
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

http_requests_total = Counter(
    "driftscope_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "driftscope_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
)

llm_requests_total = Counter(
    "driftscope_llm_requests_total",
    "LLM provider calls by outcome.",
    ["provider", "outcome"],
)

llm_request_duration_seconds = Histogram(
    "driftscope_llm_request_duration_seconds",
    "LLM provider call latency.",
    ["provider"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record count and latency for every HTTP request."""

    async def dispatch(self, request, call_next):
        # Use the matched route template ("/cases/{case_id}") rather than
        # the raw path, so one label value doesn't explode into thousands
        # of unique UUID-bearing series.
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            path = _route_template(request)
            http_requests_total.labels(request.method, path, "500").inc()
            http_request_duration_seconds.labels(request.method, path).observe(
                time.perf_counter() - started
            )
            raise

        path = _route_template(request)
        http_requests_total.labels(request.method, path, str(response.status_code)).inc()
        http_request_duration_seconds.labels(request.method, path).observe(
            time.perf_counter() - started
        )
        return response


def _route_template(request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def record_llm_call(provider: str, outcome: str, duration_seconds: float) -> None:
    """Record one LLM provider call. outcome is "success" or "error"."""
    llm_requests_total.labels(provider, outcome).inc()
    llm_request_duration_seconds.labels(provider).observe(duration_seconds)


@contextmanager
def track_llm_call(provider: str):
    """
    Wrap a provider call so both outcomes land in the metrics.

    Errors are counted and re-raised — callers keep their existing
    exception handling; this only observes.
    """
    started = time.perf_counter()
    try:
        yield
    except Exception:
        record_llm_call(provider, "error", time.perf_counter() - started)
        raise
    record_llm_call(provider, "success", time.perf_counter() - started)
