"""Retrieval observability — per-request traces and Prometheus export.

Two distinct consumers, two distinct shapes:

* **Per-request trace** (:class:`RetrievalTrace`) — everything about one query:
  the original text, what preprocessing made of it, which stages ran and how
  long each took, how many candidates survived each filter. Logged as structured
  JSON so a single slow or wrong answer can be reconstructed after the fact.

* **Aggregate metrics** (:class:`MetricsRegistry`) — counters and histograms
  across all requests, exported in Prometheus text exposition format. This is
  what a dashboard scrapes; it answers "is p95 latency climbing" rather than
  "why was this one answer bad".

The registry is deliberately a few hundred lines of plain Python rather than a
``prometheus_client`` dependency. The export format is a stable, documented text
protocol, the metric set here is small and fixed, and OpenTelemetry collectors
scrape the same endpoint — so the dependency would buy nothing but a larger
install and a second global registry to keep in sync with this one.

Thread safety: the process-wide registry is mutated from every request thread.
Counter/histogram updates take a lock; the sections are a handful of arithmetic
operations with no I/O, so contention is negligible even at high concurrency.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from logging import Logger
from typing import Any, Optional

# Latency buckets in milliseconds. Chosen around the pipeline's actual shape:
# sub-10ms is a cache hit, 50-250ms is retrieval without a reranker pass,
# 500-2500ms is the common cold path, and anything past 5s is pathological.
DEFAULT_BUCKETS_MS: tuple[float, ...] = (
    5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000,
)


# ---------------------------------------------------------------------------
# Per-request trace
# ---------------------------------------------------------------------------

@dataclass
class RetrievalTrace:
    """Structured record of one retrieval, for logging and debugging.

    Field names are stable — log aggregators key off them — so rename with
    care. Every field has a default so a trace can be emitted even when a
    request fails partway through; a partial trace is far more useful than
    none when diagnosing an error path.
    """

    # --- query understanding ---
    original_query: str = ""
    normalized_query: str = ""
    variants: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    understood: bool = False
    procedural: bool = False

    # --- stage latencies, milliseconds ---
    timings_ms: dict[str, float] = field(default_factory=dict)

    # --- candidate counts through the funnel ---
    n_bm25: int = 0
    n_vector: int = 0
    n_fused: int = 0
    n_after_rerank: int = 0
    n_after_grouping: int = 0
    n_final: int = 0
    n_images: int = 0

    # --- outcome ---
    confidence: float = 0.0
    threshold: float = 0.0
    passed_threshold: bool = False
    cache_hit: bool = False
    category: Optional[str] = None
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def log(self, logger: Logger) -> None:
        """Emit as a single structured JSON line.

        One line per request, not one per field: log aggregators index JSON
        objects, and splitting a trace across lines makes it unqueryable.
        """
        try:
            logger.info("retrieval_trace %s", json.dumps(self.to_dict(), default=str))
        except (TypeError, ValueError) as exc:  # noqa: BLE001
            # Observability must never break the request it is observing.
            logger.warning("Failed to serialise retrieval trace: %s", exc)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

class _Histogram:
    """Cumulative-bucket histogram, Prometheus semantics.

    Prometheus histograms are *cumulative*: the bucket labelled ``le="100"``
    counts every observation ≤ 100, not just those between the previous bound
    and 100. Storing per-bucket counts and summing at export time keeps the
    hot path a single increment.
    """

    def __init__(self, buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS) -> None:
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.inf_count = 0
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        self.n += 1
        self.total += value
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
                return
        self.inf_count += 1

    def cumulative(self) -> list[tuple[float, int]]:
        running = 0
        out: list[tuple[float, int]] = []
        for bound, count in zip(self.buckets, self.counts):
            running += count
            out.append((bound, running))
        return out


class MetricsRegistry:
    """Process-wide counters, gauges and histograms with Prometheus export."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = {}
        self._gauges: dict[tuple[str, tuple], float] = {}
        self._histograms: dict[tuple[str, tuple], _Histogram] = {}
        self._help: dict[str, str] = {}
        self._types: dict[str, str] = {}
        self._started = time.monotonic()

    # -- recording ----------------------------------------------------

    @staticmethod
    def _key(name: str, labels: Optional[dict[str, str]]) -> tuple[str, tuple]:
        # Labels are sorted so {a,b} and {b,a} are the same series.
        return (name, tuple(sorted((labels or {}).items())))

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict[str, str]] = None,
        help_text: str = "",
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value
            self._types[name] = "counter"
            if help_text:
                self._help[name] = help_text

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
        help_text: str = "",
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value
            self._types[name] = "gauge"
            if help_text:
                self._help[name] = help_text

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
        help_text: str = "",
        buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS,
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            hist = self._histograms.get(key)
            if hist is None:
                hist = _Histogram(buckets)
                self._histograms[key] = hist
            hist.observe(value)
            self._types[name] = "histogram"
            if help_text:
                self._help[name] = help_text

    # -- high-level: record a whole trace ------------------------------

    def record_trace(self, trace: RetrievalTrace) -> None:
        """Fold one request's trace into the aggregate metrics.

        Intent is a label rather than a separate metric name so a dashboard can
        break latency down by intent without the registry knowing the intent
        vocabulary in advance. Only the *first* intent is used: the label set
        must stay bounded, and a multi-intent query would otherwise multiply
        series count.
        """
        intent = trace.intents[0] if trace.intents else "none"

        self.counter(
            "rag_queries_total",
            labels={"intent": intent, "cache": "hit" if trace.cache_hit else "miss"},
            help_text="Total retrieval requests processed.",
        )
        self.counter(
            "rag_cache_total",
            labels={"result": "hit" if trace.cache_hit else "miss"},
            help_text="Retrieval cache lookups by outcome.",
        )
        if trace.understood:
            self.counter(
                "rag_query_understood_total",
                help_text="Queries where preprocessing detected an entity, intent, "
                          "correction or synonym expansion.",
            )
        if not trace.passed_threshold:
            self.counter(
                "rag_below_threshold_total",
                labels={"intent": intent},
                help_text="Queries whose confidence fell below the gating threshold.",
            )

        self.observe(
            "rag_request_duration_ms",
            trace.total_ms,
            labels={"intent": intent},
            help_text="End-to-end retrieval latency in milliseconds.",
        )
        for stage, ms in trace.timings_ms.items():
            self.observe(
                "rag_stage_duration_ms",
                ms,
                labels={"stage": stage},
                help_text="Per-stage retrieval latency in milliseconds.",
            )

        # Confidence is in [0,1], so it needs its own bucket set — the latency
        # buckets would put every observation in the first bucket.
        self.observe(
            "rag_confidence",
            trace.confidence,
            help_text="Retrieval confidence score.",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.observe(
            "rag_chunks_returned",
            float(trace.n_final),
            help_text="Chunks returned to the LLM after all filtering.",
            buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20),
        )

    # -- export --------------------------------------------------------

    def render_prometheus(self) -> str:
        """Render every metric in Prometheus text exposition format v0.0.4.

        This is also what OpenTelemetry collectors scrape via their Prometheus
        receiver, so one endpoint serves both.
        """
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {k: v for k, v in self._histograms.items()}
            helps = dict(self._help)
            types = dict(self._types)

        emitted: set[str] = set()

        def header(name: str) -> None:
            if name in emitted:
                return
            emitted.add(name)
            if name in helps:
                lines.append(f"# HELP {name} {helps[name]}")
            lines.append(f"# TYPE {name} {types.get(name, 'untyped')}")

        for (name, labels), value in sorted(counters.items()):
            header(name)
            lines.append(f"{name}{_fmt_labels(labels)} {value:g}")

        for (name, labels), value in sorted(gauges.items()):
            header(name)
            lines.append(f"{name}{_fmt_labels(labels)} {value:g}")

        for (name, labels), hist in sorted(histograms.items()):
            header(name)
            for bound, cumulative in hist.cumulative():
                le = _fmt_labels(labels + (("le", _fmt_float(bound)),))
                lines.append(f"{name}_bucket{le} {cumulative}")
            inf = _fmt_labels(labels + (("le", "+Inf"),))
            lines.append(f"{name}_bucket{inf} {hist.n}")
            lines.append(f"{name}_sum{_fmt_labels(labels)} {hist.total:g}")
            lines.append(f"{name}_count{_fmt_labels(labels)} {hist.n}")

        lines.append("# HELP rag_uptime_seconds Seconds since the metrics registry started.")
        lines.append("# TYPE rag_uptime_seconds gauge")
        lines.append(f"rag_uptime_seconds {time.monotonic() - self._started:g}")

        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, Any]:
        """JSON-friendly view, for the debug endpoint and tests."""
        with self._lock:
            return {
                "counters": {
                    _series_name(n, l): v for (n, l), v in self._counters.items()
                },
                "gauges": {
                    _series_name(n, l): v for (n, l), v in self._gauges.items()
                },
                "histograms": {
                    _series_name(n, l): {
                        "count": h.n,
                        "sum": h.total,
                        "mean": (h.total / h.n) if h.n else 0.0,
                    }
                    for (n, l), h in self._histograms.items()
                },
                "uptime_seconds": time.monotonic() - self._started,
            }

    def reset(self) -> None:
        """Clear all series. For tests — never call this on a live server."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


def _fmt_float(value: float) -> str:
    # Prometheus bucket bounds are conventionally rendered without a trailing
    # ".0" when integral, which keeps series names stable across restarts.
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _fmt_labels(labels: tuple) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(str(v))}"' for k, v in labels)
    return "{" + inner + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _series_name(name: str, labels: tuple) -> str:
    return f"{name}{_fmt_labels(labels)}" if labels else name


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_registry = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _registry
