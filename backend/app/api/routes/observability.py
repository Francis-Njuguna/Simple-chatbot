"""Metrics endpoint for Prometheus scraping and debugging."""

from typing import Any

from fastapi import APIRouter, Response

from backend.app.database.session import pool_stats, publish_pool_gauges
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
    # Sampled here so the exported gauge reflects the pool at scrape time.
    publish_pool_gauges()
    text = get_metrics().render_prometheus()
    return Response(content=text, media_type="text/plain; version=0.0.4")


@router.get("/metrics/debug")
async def metrics_debug() -> dict[str, Any]:
    """JSON debug view of all metrics plus cache and DB-pool stats.

    For development and troubleshooting. Not scraped; use `/metrics` for that.

    ``db_pool`` is the live pool snapshot — sizes and counts only. It carries no
    DSN, host or credential, so it is safe to read from a browser while a load
    test runs. Pair it with the ``db_connection_acquire_ms`` histogram: rising
    acquire times with ``available`` at 0 is pool exhaustion, whereas a healthy
    pool shows sub-millisecond acquisition regardless of how busy the app is.
    """
    snapshot = get_metrics().snapshot()
    retriever = get_retriever()
    snapshot["cache"] = retriever.cache_stats()
    snapshot["db_pool"] = pool_stats()
    return snapshot
