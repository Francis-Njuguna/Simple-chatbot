"""NVIDIA-only latency diagnostic for isolated generation and full RAG.

This script is intentionally diagnostic-only: it reads the current settings,
uses the existing RAG components, and writes reports without changing them.
The isolated phase calls NVIDIA NIM directly with a fixed context; the RAG
phase mirrors RAGService's stage order and records the existing retriever trace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

QUERIES = [
    "How do I set up Microsoft Authenticator?",
    "I changed phones and Authenticator is blocking my login.",
    "How can I enroll for 2FA on Office 365?",
    "How do I log in to the LMS?",
    "How do I log into Moodle?",
    "I cant acces moddle, how do I sine in?",
    "My Moodle assignments are not opening.",
    "How do I reset my student portal password?",
    "My Student Portal login is not working.",
    "I forgot my password and am locked out.",
    "How do I register for and take a VAS exam?",
    "Where are the VAS training slides?",
    "What is SMOWL proctoring?",
    "SMOWL camera is not working during my exam.",
    "How do I access my university email?",
    "Where do I sign in to corporate Outlook?",
    "My laptop cannot connect to the campus Wi-Fi.",
    "The portal shows an error when I upload a document.",
    "Hi, can you help me contact the Amref Help Desk?",
    "Please help me log in.",
]

FIXED_CONTEXT = """AMREF Help Desk reference material:

Student Portal: choose Forgot password on the student portal sign-in page,
enter the registered university email, and follow the reset link. Check junk
mail if the message does not arrive and contact ICT Help Desk if needed.

Moodle/LMS: sign in with the university account. Moodle is the learning
management system used for assignments and course material. Login failures
should be reported to the Help Desk after checking the account password.

Microsoft Authenticator and Office 365: install Microsoft Authenticator, add
the work account, scan the QR code shown during enrollment, and approve the
sign-in notification. A changed phone may require ICT to reset MFA.

VAS and SMOWL: VAS is the Virtual Assessment System used for exams. SMOWL
provides webcam monitoring; allow camera access and run its compatibility
check before an exam. Contact the Help Desk for exam access problems.

University email and IT support: use the university Outlook account. For Wi-Fi,
upload, or device errors, record the exact message, restart the device, and
contact the AMREF ICT Help Desk with a screenshot and account details.
""".strip()

FAILURE_MARKERS = (
    "could not generate an answer:",
    "did not start answering within",
    "the answer was cut off:",
    "handling other requests",
)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def redacted_config(settings: Any, llm: Any) -> dict[str, Any]:
    info = llm.describe()
    return {
        "provider": "NVIDIA NIM (OpenAI-compatible transport)",
        "model": info.get("model"),
        "api_base": str(settings.openai_api_base or ""),
        "llm_timeout_s": settings.llm_timeout,
        "first_token_timeout_s": settings.llm_first_token_timeout,
        "stream_stall_timeout_s": settings.llm_stream_stall_timeout,
        "max_output_tokens": info.get("max_tokens"),
        "temperature": info.get("temperature"),
        "max_retries": settings.llm_max_retries,
        "llm_max_concurrency": settings.llm_max_concurrency,
        "llm_queue_timeout_s": settings.llm_queue_timeout,
        "http_max_connections": info.get("http_max_connections"),
        "http_max_keepalive": info.get("http_max_keepalive"),
        "streaming_path": "LLMService.stream_answer / NVIDIA SSE",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model if settings.embedding_provider != "ollama" else settings.ollama_embedding_model,
        "embedding_device": settings.embedding_device,
        "rerank_enabled": settings.rerank_enabled,
        "rerank_model": settings.rerank_model,
        "rerank_quantize": settings.rerank_quantize,
        "rerank_shortlist": settings.rerank_shortlist,
        "rerank_query_forms": settings.rerank_query_forms,
        "top_k_retrieval": settings.top_k_retrieval,
        "db_pool_size": settings.db_pool_size,
        "db_max_overflow": settings.db_max_overflow,
        "db_pool_timeout_s": settings.db_pool_timeout,
        "chroma_mode": settings.chroma_mode,
        "chroma_persist_dir": settings.chroma_persist_dir,
    }


async def isolated_stream(client: Any, model: str, base_url: str, key: str, question: str, *, temperature: Any, max_tokens: int) -> dict[str, Any]:
    """One raw SSE request; headers and visible-token clocks are separate."""
    import httpx

    prompt = f"Answer the AMREF Help Desk question using only the reference material.\n\nReference:\n{FIXED_CONTEXT}\n\nQuestion: {question}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are the AMREF Help Desk assistant. Be concise and grounded."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    started = time.perf_counter()
    first_headers: float | None = None
    first_frame: float | None = None
    first_visible: float | None = None
    reasoning_chars = 0
    answer_chars = 0
    chunks = 0
    usage: dict[str, Any] = {}
    error = ""
    statuses: list[int] = []
    try:
        async with client.stream(
            "POST", f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=payload,
        ) as response:
            first_headers = time.perf_counter() - started
            statuses.append(response.status_code)
            if response.status_code >= 400:
                error = f"HTTP {response.status_code}: {(await response.aread()).decode('utf-8', 'replace')[:240]}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    now = time.perf_counter() - started
                    if first_frame is None:
                        first_frame = now
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []) or []:
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning_content") or ""
                        content = delta.get("content") or ""
                        reasoning_chars += len(reasoning)
                        answer_chars += len(content)
                        if content:
                            chunks += 1
                            if first_visible is None:
                                first_visible = now
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - started
    return {
        "question": question,
        "ok": not error and answer_chars > 0,
        "error": error,
        "http_statuses": statuses,
        "connection_to_headers_s": first_headers,
        "first_frame_s": first_frame,
        "ttft_after_headers_s": (first_visible - first_headers) if first_visible is not None and first_headers is not None else None,
        "visible_from_request_s": first_visible,
        "post_first_visible_to_completion_s": (total - first_visible) if first_visible is not None else None,
        "total_generation_s": total,
        "answer_chars": answer_chars,
        "reasoning_chars": reasoning_chars,
        "reasoning_tokens_observed": reasoning_chars > 0,
        "chunks": chunks,
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "input_tokens_est": round(len(prompt) / 4),
        "tokens_per_second": ((usage.get("completion_tokens") or usage.get("output_tokens")) / total if usage.get("completion_tokens") or usage.get("output_tokens") else None),
    }


async def run_isolated(settings: Any, llm: Any, levels: list[int], requests_per_level: int) -> dict[str, Any]:
    import httpx

    key = (settings.openai_api_key or "").strip()
    if not key:
        raise RuntimeError("NVIDIA credential is not configured")
    results: dict[str, Any] = {}
    limits = httpx.Limits(max_connections=max(levels), max_keepalive_connections=max(levels))
    timeout = httpx.Timeout(connect=settings.llm_connect_timeout, read=max(90.0, float(settings.llm_timeout) * 4), write=90.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for level in levels:
            sem = asyncio.Semaphore(level)
            started = time.perf_counter()

            async def one(index: int) -> dict[str, Any]:
                async with sem:
                    return await isolated_stream(
                        client, settings.openai_model, settings.openai_api_base or "", key,
                        QUERIES[index % len(QUERIES)],
                        temperature=getattr(llm._llm, "temperature", None),
                        max_tokens=settings.llm_max_tokens,
                    )

            rows = await asyncio.gather(*(one(i) for i in range(requests_per_level)))
            wall = time.perf_counter() - started
            good = [r for r in rows if r["ok"]]
            results[str(level)] = {
                "concurrency": level,
                "requests": len(rows),
                "successes": len(good),
                "failures": len(rows) - len(good),
                "success_rate": len(good) / len(rows) if rows else 0.0,
                "timeout_rate": sum("timeout" in r["error"].lower() for r in rows) / len(rows) if rows else 0.0,
                "wall_s": wall,
                "throughput_answers_per_s": len(good) / wall if wall else 0.0,
                "connection_to_headers_s": stats([r["connection_to_headers_s"] for r in good if r["connection_to_headers_s"] is not None]),
                "ttft_after_headers_s": stats([r["ttft_after_headers_s"] for r in good if r["ttft_after_headers_s"] is not None]),
                "visible_from_request_s": stats([r["visible_from_request_s"] for r in good if r["visible_from_request_s"] is not None]),
                "post_first_visible_to_completion_s": stats([r["post_first_visible_to_completion_s"] for r in good if r["post_first_visible_to_completion_s"] is not None]),
                "total_generation_s": stats([r["total_generation_s"] for r in good]),
                "output_tokens": stats([float(r["output_tokens"]) for r in good if r["output_tokens"] is not None]),
                "input_tokens": stats([float(r["input_tokens"]) for r in good if r["input_tokens"] is not None]),
                "input_tokens_est": stats([float(r["input_tokens_est"]) for r in rows]),
                "tokens_per_second": stats([float(r["tokens_per_second"]) for r in good if r["tokens_per_second"] is not None]),
                "reasoning_tokens_observed": sum(r["reasoning_tokens_observed"] for r in rows),
                "http_statuses": dict(Counter(str(s) for r in rows for s in r["http_statuses"])),
                "errors": dict(Counter(r["error"][:180] or "unknown" for r in rows if not r["ok"])),
                "observations": rows,
            }
            print(f"isolated concurrency={level}: {len(good)}/{len(rows)} ok, wall={wall:.1f}s", flush=True)
    return results


async def run_rag(settings: Any, retriever: Any, llm: Any, requests: int) -> dict[str, Any]:
    from backend.app.database.models import AnalyticsLog, ChatMessage
    from backend.app.database.session import db_scope
    from backend.app.services.rag_service import RAGService

    rag = RAGService(retriever=retriever, llm_service=llm)
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(QUERIES[:requests]):
        started = time.perf_counter()
        row: dict[str, Any] = {"question": question, "query_index": index + 1}
        # Mirror RAGService._prepare, while retaining the retriever trace.
        db_start = time.perf_counter()
        async with db_scope("benchmark_session_history") as db:
            session_id = await rag._get_or_create_session(db, None, question)
            history = await rag._get_history_text(db, session_id)
            db.add(ChatMessage(session_id=session_id, role="user", content=question))
            await db.flush()
        row["session_history_ms"] = (time.perf_counter() - db_start) * 1000

        t = time.perf_counter()
        embedding = await retriever.embed_query(question)
        row["embedding_ms"] = (time.perf_counter() - t) * 1000

        traces: list[Any] = []
        t = time.perf_counter()
        chunks, images, processed = await retriever.retrieve(question, query_embedding=embedding, trace_sink=traces)
        row["retrieval_wrapper_ms"] = (time.perf_counter() - t) * 1000
        trace = traces[-1] if traces else None
        row["retrieval_trace_ms"] = dict(trace.timings_ms) if trace else {}
        row["retrieval_counts"] = {k: getattr(trace, k) for k in ("n_bm25", "n_vector", "n_fused", "n_after_rerank", "n_final", "n_images") if trace is not None}

        t = time.perf_counter()
        async with db_scope("benchmark_hydration") as db:
            chunks, images = await retriever.hydrate_results(db, chunks, images)
        row["hydration_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        context = retriever.format_context(chunks)
        image_context = retriever.format_images(images)
        row["context_build_ms"] = (time.perf_counter() - t) * 1000
        row["context_chars"] = len(context)

        t = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        reasoning = 0
        async for token in llm.stream_answer(question, context, history, image_context):
            if first is None and token:
                first = time.perf_counter() - t
            parts.append(token)
        row["llm_generation_ms"] = (time.perf_counter() - t) * 1000
        row["llm_ttft_ms"] = first * 1000 if first is not None else None
        row["answer"] = "".join(parts)
        row["llm_ok"] = not any(m in row["answer"].lower() for m in FAILURE_MARKERS)
        row["answer_chars"] = len(row["answer"])
        row["reasoning_tokens_observed"] = reasoning > 0

        t = time.perf_counter()
        async with db_scope("benchmark_persist") as db:
            metadata = {"sources": [], "images": [], "confidence": retriever.compute_confidence(chunks, processed)}
            db.add(ChatMessage(session_id=session_id, role="assistant", content=row["answer"], metadata_=metadata))
            db.add(AnalyticsLog(event_type="benchmark_chat_query", session_id=session_id, payload={"message": question[:200]}))
            await db.flush()
        row["persist_ms"] = (time.perf_counter() - t) * 1000
        row["total_end_to_end_ms"] = (time.perf_counter() - started) * 1000
        rows.append(row)
        print(f"rag {index + 1:02d}/{requests}: total={row['total_end_to_end_ms']/1000:.1f}s llm={row['llm_generation_ms']/1000:.1f}s retrieval={row['retrieval_wrapper_ms']/1000:.2f}s", flush=True)
    return {"requests": len(rows), "observations": rows}


def aggregate_rag(rag: dict[str, Any]) -> dict[str, Any]:
    rows = rag["observations"]
    fields = ["session_history_ms", "embedding_ms", "retrieval_wrapper_ms", "hydration_ms", "context_build_ms", "llm_generation_ms", "persist_ms", "total_end_to_end_ms"]
    out: dict[str, Any] = {f: stats([float(r[f]) for r in rows]) for f in fields}
    trace_fields = sorted({k for r in rows for k in r.get("retrieval_trace_ms", {})})
    out["retrieval_trace"] = {f: stats([float(r["retrieval_trace_ms"][f]) for r in rows if f in r.get("retrieval_trace_ms", {})]) for f in trace_fields}
    out["llm_success_rate"] = sum(bool(r["llm_ok"]) for r in rows) / len(rows) if rows else 0.0
    total_median = out["total_end_to_end_ms"]["median"] or 0.0
    out["latency_share_percent_median"] = {f: ((out[f]["median"] or 0.0) / total_median * 100.0 if total_median else 0.0) for f in fields if f != "total_end_to_end_ms"}
    for f in trace_fields:
        out["latency_share_percent_median"][f] = ((out["retrieval_trace"][f]["median"] or 0.0) / total_median * 100.0 if total_median else 0.0)
    return out


def markdown(payload: dict[str, Any]) -> str:
    cfg = payload["configuration"]
    iso = payload["isolated"]
    rag = payload.get("rag", {})
    a = rag.get("aggregate", {})
    lines = [
        "# NVIDIA NIM LLM Performance Diagnostic",
        "",
        f"Timestamp: {payload['metadata']['timestamp_utc']}",
        "",
        "## 1. Current NVIDIA configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for k, v in cfg.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += ["", "## 2. Isolated LLM benchmark", "", f"Fixed context; {len(QUERIES)} representative queries; direct NVIDIA SSE; no retrieval.", "", "| Concurrency | Requests | Success | TTFT after headers p50 | Total generation p50 | Output tokens p50 | Tok/s p50 | Throughput |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level, r in iso.items():
        f = lambda name: "n/a" if r[name]["median"] is None else f"{r[name]['median']:.2f}s"
        tok = "n/a" if r["output_tokens"]["median"] is None else f"{r['output_tokens']['median']:.0f}"
        tps = "n/a" if r["tokens_per_second"]["median"] is None else f"{r['tokens_per_second']['median']:.1f}"
        lines.append(f"| {level} | {r['requests']} | {r['success_rate']:.1%} | {f('ttft_after_headers_s')} | {f('total_generation_s')} | {tok} | {tps} | {r['throughput_answers_per_s']:.3f}/s |")
    lines += ["", "## 3. Concurrency results", "", "Concurrency was run against NVIDIA directly, with 20 queries per level. HTTP errors, timeouts, and provider quota responses are retained in the JSON observations.", "", "## 4. Full RAG latency breakdown", ""]
    if a:
        lines += ["| Stage | Median | p95 | Share of median end-to-end |", "|---|---:|---:|---:|"]
        for f in ("session_history_ms", "embedding_ms", "retrieval_wrapper_ms", "hydration_ms", "context_build_ms", "llm_generation_ms", "persist_ms", "total_end_to_end_ms"):
            if f in a:
                share = "-" if f == "total_end_to_end_ms" else f"{a['latency_share_percent_median'].get(f, 0):.1f}%"
                lines.append(f"| `{f}` | {a[f]['median']:.0f} ms | {a[f]['p95']:.0f} ms | {share} |")
        lines += ["", "### Retrieval sub-stages", "", "| Stage | Median | p95 |", "|---|---:|---:|"]
        for f, v in a.get("retrieval_trace", {}).items():
            lines.append(f"| `{f}` | {v['median']:.0f} ms | {v['p95']:.0f} ms |")
    else:
        lines.append("Full RAG phase did not complete; see `rag.error` in JSON.")
    lines += ["", "## 5. Streaming analysis", "", "`connection_to_headers_s` is measured separately from `ttft_after_headers_s`; TTFT excludes the request-to-headers interval. `reasoning_content` is counted from raw SSE deltas when NVIDIA returns it. The app's `LLMService.stream_answer` yields only answer text, so reasoning deltas are not shown to students.", "", "## 6. Error and timeout analysis", "", "| Concurrency | Success | Timeout rate | HTTP statuses | Errors |", "|---:|---:|---:|---|---|"]
    for level, r in iso.items():
        statuses = ", ".join(f"{k} x{v}" for k, v in r["http_statuses"].items()) or "none"
        errors = ", ".join(f"{k} x{v}" for k, v in r["errors"].items()) or "none"
        lines.append(f"| {level} | {r['successes']}/{r['requests']} | {r['timeout_rate']:.1%} | {statuses} | {errors} |")
    if a:
        lines += ["", f"Full RAG answer success rate: **{a.get('llm_success_rate', 0.0):.1%}**. Queue-status prose is not counted as an isolated answer."]
    else:
        lines += ["", "Full RAG measurements were unavailable. Queue-status prose is not counted as an isolated answer."]
    lines += ["", "## 7. Bottleneck identification", ""]
    if a:
        llm_share = a["latency_share_percent_median"].get("llm_generation_ms", 0.0)
        lines.append(f"The median NVIDIA generation share of end-to-end RAG latency was **{llm_share:.1f}%**; retrieval wrapper share was **{a['latency_share_percent_median'].get('retrieval_wrapper_ms', 0.0):.1f}%**. The database-scoped stages (session/history, hydration, and persistence) together consumed **{sum(a['latency_share_percent_median'].get(k, 0.0) for k in ('session_history_ms', 'hydration_ms', 'persist_ms')):.1f}%** of median end-to-end latency.")
    else:
        lines.append("No complete RAG measurement was available.")
    lines += ["", "## 8. Theoretical <10s calculation", ""]
    if a:
        healthy_retrieval = sum((a[f]["median"] or 0.0) for f in ("session_history_ms", "embedding_ms", "retrieval_wrapper_ms", "hydration_ms", "context_build_ms", "persist_ms"))
        llm_med = a["llm_generation_ms"]["median"] or 0.0
        lines.append(f"Measured median non-LLM stages: **{healthy_retrieval/1000:.2f}s**; measured median NVIDIA generation: **{llm_med/1000:.2f}s**; theoretical best-case with those healthy medians retained: **{(healthy_retrieval + llm_med)/1000:.2f}s** before network/client variance.")
        lines.append("The target is realistically achievable only if this best-case value is below 10s and p95 is separately controlled; median alone does not guarantee the target for every request.")
    else:
        lines.append("Cannot calculate until full RAG measurements complete.")
    lines += ["", "## 9. Recommended optimizations ranked by expected impact", "", f"1. Reduce NVIDIA answer-generation time first: it measured {a.get('latency_share_percent_median', {}).get('llm_generation_ms', 0.0):.1f}% of median end-to-end latency, and the measured median was {((a.get('llm_generation_ms', {}).get('median') or 0.0) / 1000):.2f}s.", f"2. Eliminate avoidable provider failures before raising concurrency: the isolated run reached {max((r['success_rate'] for r in iso.values()), default=0.0):.1%} at its best level but fell to {iso.get('10', {}).get('success_rate', 0.0):.1%} at concurrency 10 because of HTTP 429 responses.", f"3. Investigate persistence and database variance next: the measured persistence median was {((a.get('persist_ms', {}).get('median') or 0.0) / 1000):.2f}s and p95 was {((a.get('persist_ms', {}).get('p95') or 0.0) / 1000):.2f}s.", f"4. Profile reranking only after the NVIDIA path: reranking measured {((a.get('retrieval_trace', {}).get('rerank', {}).get('median') or 0.0) / 1000):.2f}s median, materially smaller than NVIDIA generation but still the largest CPU retrieval sub-stage.", "", "## Blunt conclusion", ""]
    if a:
        llm_share = a["latency_share_percent_median"].get("llm_generation_ms", 0.0)
        lines.append(f"**NVIDIA LLM is {'YES' if llm_share >= 50 else 'NO'} the main median-latency bottleneck** in this run. Its measured contribution is approximately **{llm_share:.1f}%** of median end-to-end latency.")
    else:
        lines.append("**Undetermined:** the full RAG phase did not produce measurements.")
    return "\n".join(lines) + "\n"


async def main(args: argparse.Namespace) -> int:
    from backend.app.config import get_settings
    from backend.app.rag.llm import get_llm_service
    from backend.app.rag.retriever import get_retriever

    settings = get_settings()
    if settings.llm_provider != "openai" or "integrate.api.nvidia.com" not in (settings.openai_api_base or ""):
        raise RuntimeError("This diagnostic requires the configured NVIDIA NIM provider only")
    llm = get_llm_service()
    retriever = get_retriever()
    print(f"NVIDIA model={settings.openai_model}; endpoint={settings.openai_api_base}; credential present (not displayed)", flush=True)
    # Heavy local components are warmed once and excluded from per-query rows.
    await retriever.embed_query("benchmark warmup")
    from backend.app.rag.reranker import get_reranker
    get_reranker().warmup()

    payload: dict[str, Any] = {
        "metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python_version": sys.version, "platform": platform.platform(), "hostname": platform.node()},
        "configuration": redacted_config(settings, llm),
        "queries": QUERIES,
        "fixed_context_chars": len(FIXED_CONTEXT),
    }
    payload["isolated"] = await run_isolated(settings, llm, args.levels, args.requests_per_level)
    try:
        payload["rag"] = await run_rag(settings, retriever, llm, len(QUERIES))
        payload["rag"]["aggregate"] = aggregate_rag(payload["rag"])
    except Exception as exc:  # preserve isolated results and explain infra blockers
        payload["rag"] = {"error": f"{type(exc).__name__}: {exc}", "observations": [], "aggregate": {}}
        print(f"RAG phase failed: {payload['rag']['error']}", file=sys.stderr, flush=True)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(markdown(payload), encoding="utf-8")
    print(f"wrote {output_json} and {output_md}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,2,5,10")
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--output-json", default="benchmarks/nvidia_llm_benchmark.json")
    parser.add_argument("--output-md", default="benchmarks/nvidia_llm_benchmark.md")
    args = parser.parse_args()
    args.levels = [int(x) for x in args.levels.split(",") if x.strip()]
    raise SystemExit(asyncio.run(main(args)))
