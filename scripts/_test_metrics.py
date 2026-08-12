"""Verify the metrics registry and Prometheus exposition format."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.utils.metrics import MetricsRegistry, RetrievalTrace


def test_counters_and_labels():
    m = MetricsRegistry()
    m.counter("rag_queries_total", labels={"intent": "password_reset"}, help_text="Total.")
    m.counter("rag_queries_total", labels={"intent": "password_reset"})
    m.counter("rag_queries_total", labels={"intent": "email_access"})

    snap = m.snapshot()["counters"]
    assert snap['rag_queries_total{intent="password_reset"}'] == 2.0
    assert snap['rag_queries_total{intent="email_access"}'] == 1.0

    # Label order must not create separate series
    m2 = MetricsRegistry()
    m2.counter("x", labels={"a": "1", "b": "2"})
    m2.counter("x", labels={"b": "2", "a": "1"})
    assert list(m2.snapshot()["counters"].values()) == [2.0]
    print("[ok] counters + label normalisation")


def test_histogram_cumulative():
    m = MetricsRegistry()
    for v in (3, 7, 30, 300, 99999):
        m.observe("lat_ms", v)

    text = m.render_prometheus()
    # Cumulative semantics: le="10" must include the 3 and the 7.
    assert 'lat_ms_bucket{le="5"} 1' in text, text
    assert 'lat_ms_bucket{le="10"} 2' in text, text
    assert 'lat_ms_bucket{le="50"} 3' in text, text
    assert 'lat_ms_bucket{le="500"} 4' in text, text
    assert 'lat_ms_bucket{le="+Inf"} 5' in text, text
    assert "lat_ms_count 5" in text, text
    print("[ok] histogram cumulative buckets")


def test_prometheus_format():
    m = MetricsRegistry()
    m.counter("c_total", 5, help_text="A counter.")
    m.gauge("g_now", 1.5, help_text="A gauge.")
    text = m.render_prometheus()

    assert "# HELP c_total A counter." in text
    assert "# TYPE c_total counter" in text
    assert "c_total 5" in text
    assert "# TYPE g_now gauge" in text
    assert "g_now 1.5" in text
    assert "rag_uptime_seconds" in text

    # HELP/TYPE emitted exactly once per metric name even with many series
    m2 = MetricsRegistry()
    for i in range(5):
        m2.counter("multi", labels={"i": str(i)}, help_text="H")
    t2 = m2.render_prometheus()
    assert t2.count("# TYPE multi counter") == 1, t2
    print("[ok] prometheus exposition format")


def test_label_escaping():
    m = MetricsRegistry()
    m.counter("esc", labels={"q": 'he said "hi"\\ok'})
    text = m.render_prometheus()
    assert r'\"hi\"' in text and r"\\ok" in text, text
    print("[ok] label escaping")


def test_record_trace():
    m = MetricsRegistry()
    trace = RetrievalTrace(
        original_query="moddle login",
        normalized_query="moodle login",
        variants=["moodle login", "lms login"],
        intents=["login_trouble"],
        entities=["Moodle"],
        corrections={"moddle": "moodle"},
        understood=True,
        procedural=True,
        timings_ms={"bm25": 4.2, "vector": 31.0, "rerank": 210.5},
        n_bm25=20, n_vector=20, n_fused=30,
        n_after_rerank=8, n_after_grouping=5, n_final=5,
        confidence=0.61, threshold=0.30, passed_threshold=True,
        cache_hit=False, total_ms=280.0,
    )
    m.record_trace(trace)
    text = m.render_prometheus()

    assert 'rag_queries_total{cache="miss",intent="login_trouble"} 1' in text, text
    assert 'rag_cache_total{result="miss"} 1' in text, text
    assert "rag_query_understood_total 1" in text, text
    assert 'rag_stage_duration_ms_bucket{stage="rerank"' in text, text
    assert "rag_confidence_count 1" in text, text
    assert "rag_below_threshold_total" not in text, "passed queries must not count as below-threshold"

    # A failing query should register below-threshold
    m.record_trace(RetrievalTrace(intents=["definition"], passed_threshold=False))
    assert 'rag_below_threshold_total{intent="definition"} 1' in m.render_prometheus()
    print("[ok] record_trace end-to-end")


def test_trace_serialisation():
    t = RetrievalTrace(original_query="test", intents=["a"])
    d = t.to_dict()
    assert d["original_query"] == "test"
    assert d["intents"] == ["a"]
    assert d["n_final"] == 0
    print("[ok] trace serialisation")


if __name__ == "__main__":
    test_counters_and_labels()
    test_histogram_cumulative()
    test_prometheus_format()
    test_label_escaping()
    test_record_trace()
    test_trace_serialisation()
    print("\nAll metrics tests passed")
