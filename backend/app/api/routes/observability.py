"""Metrics endpoint for Prometheus scraping and debugging."""

from typing import Any

from fastapi import APIRouter, Response

from backend.app.utils.metrics import get_metrics
from backend.app.rag.retriever import get_retriever

router = APIRouter(tags=["observability"])


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus text exposition format endpoint.

    Scraped by Prometheus, Grafana Agent, or OpenTelemetry collectors via the
    Prometheus receiver. Returns cumulative counters and histograms covering
    query volume, latency, confidence, cache effectiveness, and per-stage
    breakdowns.
    """
    text = get_metrics().render_prometheus()
    return Response(content=text, media_type="text/plain; version=0.0.4")


@router.get("/metrics/debug")
async def metrics_debug() -> dict[str, Any]:
    """JSON debug view of all metrics plus cache stats.

    For development and troubleshooting. Not scraped; use `/metrics` for that.
    """
    snapshot = get_metrics().snapshot()
    retriever = get_retriever()
    snapshot["cache"] = retriever.cache_stats()
    return snapshot
