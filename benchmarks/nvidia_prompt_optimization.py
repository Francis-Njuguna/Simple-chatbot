"""A/B benchmark for the legacy and compact NVIDIA production prompts.

This benchmark uses the real retrieval and persistence path with the same
representative queries for both profiles. It intentionally runs one request at
a time by default so provider throttling does not hide prompt effects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.benchmark_nvidia_diagnostic import FAILURE_MARKERS, QUERIES


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def metric(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 95),
        "mean": statistics.mean(values) if values else None,
    }


def token_count(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return round(len(text) / 4)


def quality_proxy(question: str, answer: str, context: str) -> dict[str, Any]:
    lower_question = question.lower()
    lower_answer = answer.lower()
    expected = {
        "mfa": ("authenticator", "mfa", "2fa"),
        "authenticator": ("authenticator", "approve", "sign-in"),
        "moodle": ("moodle", "lms", "login"),
        "student portal": ("student portal", "password", "portal"),
        "smowl": ("smowl", "camera", "exam"),
        "vas": ("vas", "exam", "assessment"),
        "email": ("email", "outlook", "university"),
    }
    terms: tuple[str, ...] = ()
    for key, candidates in expected.items():
        if key in lower_question:
            terms = candidates
            break
    relevant = bool(terms) and any(term in lower_answer for term in terms)
    context_urls = set(__import__("re").findall(r"https?://[^\s)]+", context))
    answer_urls = set(__import__("re").findall(r"https?://[^\s)]+", answer))
    hallucinated_urls = sorted(answer_urls - context_urls)
    failure = any(marker in lower_answer for marker in FAILURE_MARKERS)
    return {
        "relevance": relevant and not failure,
        "groundedness": not hallucinated_urls and not failure,
        "completeness_proxy": len(answer.strip()) >= 80 and not failure,
        "hallucinated_urls": hallucinated_urls,
        "short_query_handled": len(question.split()) <= 3 and not failure and relevant,
        "answer_chars": len(answer),
    }


async def build_service(profile: str) -> Any:
    from backend.app.config import get_settings
    from backend.app.rag.llm import LLMService

    base = get_settings()
    if base.llm_provider != "openai" or "integrate.api.nvidia.com" not in (base.openai_api_base or ""):
        raise RuntimeError("This benchmark requires the configured NVIDIA NIM provider only")
    settings = base.model_copy(update={"llm_prompt_profile": profile})
    service = LLMService.__new__(LLMService)
    service.settings = settings
    service._llm = service._build_llm()
    service._generation_gate = None
    return service


async def run_profile(profile: str, requests: int) -> dict[str, Any]:
    from backend.app.database.models import AnalyticsLog, ChatMessage
    from backend.app.database.session import db_scope
    from backend.app.rag.retriever import get_retriever
    from backend.app.services.rag_service import RAGService

    llm = await build_service(profile)
    retriever = get_retriever()
    rag = RAGService(retriever=retriever, llm_service=llm)
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(QUERIES[:requests]):
        started = time.perf_counter()
        row: dict[str, Any] = {"question": question, "query_index": index + 1, "profile": profile}
        async with db_scope("prompt_ab_session") as db:
            session_id = await rag._get_or_create_session(db, None, question)
            history = await rag._get_history_text(db, session_id)
            db.add(ChatMessage(session_id=session_id, role="user", content=question))
            await db.flush()

        t = time.perf_counter()
        embedding = await retriever.embed_query(question)
        row["embedding_ms"] = (time.perf_counter() - t) * 1000
        traces: list[Any] = []
        t = time.perf_counter()
        chunks, images, processed = await retriever.retrieve(question, query_embedding=embedding, trace_sink=traces)
        row["retrieval_ms"] = (time.perf_counter() - t) * 1000
        trace = traces[-1] if traces else None
        row["retrieval_counts"] = {k: getattr(trace, k) for k in ("n_bm25", "n_vector", "n_fused", "n_after_rerank", "n_final") if trace is not None}
        t = time.perf_counter()
        async with db_scope("prompt_ab_hydration") as db:
            chunks, images = await retriever.hydrate_results(db, chunks, images)
        row["hydration_ms"] = (time.perf_counter() - t) * 1000
        context = retriever.format_context(chunks)
        image_context = retriever.format_images(images)
        messages = llm._build_messages(question, context, history, image_context)
        row["context_chars"] = len(context)
        row["prompt_chars"] = sum(len(str(message.content)) for message in messages)
        row["input_tokens_est"] = token_count("\n".join(str(message.content) for message in messages))
        row["system_prompt_tokens_est"] = token_count(str(messages[0].content))
        row["user_prompt_tokens_est"] = token_count(str(messages[1].content))

        stream_stats: dict[str, Any] = {}
        parts: list[str] = []
        async for token in llm.stream_answer(question, context, history, image_context, stats=stream_stats):
            parts.append(token)
        answer = "".join(parts)
        row.update({
            "answer": answer,
            "answer_chars": len(answer),
            "llm_generation_ms": stream_stats.get("llm_generation_ms"),
            "ttft_ms": stream_stats.get("ttft_ms"),
            "input_tokens": stream_stats.get("input_tokens"),
            "output_tokens": stream_stats.get("output_tokens"),
            "reasoning_chars": stream_stats.get("reasoning_chars", 0),
            "llm_ok": bool(stream_stats.get("ok")) and not any(marker in answer.lower() for marker in FAILURE_MARKERS),
            "llm_error": stream_stats.get("error", ""),
        })
        t = time.perf_counter()
        async with db_scope("prompt_ab_persist") as db:
            db.add(ChatMessage(session_id=session_id, role="assistant", content=answer, metadata_={"sources": [], "images": [], "confidence": retriever.compute_confidence(chunks, processed)}))
            db.add(AnalyticsLog(event_type="prompt_ab", session_id=session_id, payload={"message": question[:200], "profile": profile}))
            await db.flush()
        row["persistence_ms"] = (time.perf_counter() - t) * 1000
        row["total_ms"] = (time.perf_counter() - started) * 1000
        row["quality"] = quality_proxy(question, answer, context)
        rows.append(row)
        print(f"{profile} {index + 1:02d}/{requests}: total={row['total_ms'] / 1000:.2f}s llm={(row.get('llm_generation_ms') or 0) / 1000:.2f}s", flush=True)

    # Keep the provider transport alive across the two A/B variants. The
    # OpenAI-compatible client may share its underlying async pool, and closing
    # one variant here can invalidate the next variant's requests.
    return {"profile": profile, "requests": len(rows), "observations": rows}


def aggregate(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["observations"]
    fields = ("total_ms", "llm_generation_ms", "ttft_ms", "embedding_ms", "retrieval_ms", "persistence_ms", "context_chars", "prompt_chars", "input_tokens_est", "system_prompt_tokens_est", "user_prompt_tokens_est", "input_tokens", "output_tokens", "answer_chars", "reasoning_chars")
    out = {field: metric([float(row[field]) for row in rows if row.get(field) is not None]) for field in fields}
    out["success_rate"] = sum(bool(row.get("llm_ok")) for row in rows) / len(rows) if rows else 0.0
    out["rate_429"] = sum("429" in str(row.get("llm_error", "")) for row in rows) / len(rows) if rows else 0.0
    out["quality"] = {
        key: sum(bool(row["quality"].get(key)) for row in rows) / len(rows) if rows else 0.0
        for key in ("relevance", "groundedness", "completeness_proxy", "short_query_handled")
    }
    out["errors"] = dict(Counter(str(row.get("llm_error")) for row in rows if row.get("llm_error")))
    return out


def render(payload: dict[str, Any]) -> str:
    a = payload["variants"]["legacy"]["aggregate"]
    b = payload["variants"]["compact"]["aggregate"]
    def value(data: dict[str, Any], field: str, key: str = "median") -> str:
        item = data.get(field, {})
        raw = item.get(key)
        return "n/a" if raw is None else f"{raw / 1000:.2f}s" if field.endswith("_ms") else f"{raw:.0f}"
    def change(field: str) -> str:
        old, new = a.get(field, {}).get("median"), b.get(field, {}).get("median")
        return "n/a" if old in (None, 0) or new is None else f"{(new - old) / old * 100:+.1f}%"
    lines = ["# NVIDIA Prompt Optimization A/B", "", "This report compares the same representative queries through the real retrieval, NVIDIA streaming, and persistence path. Quality values are automated proxies, not a substitute for human review.", "", "## Configuration", "", f"- Model: `{payload['configuration']['model']}`", f"- NVIDIA base URL: `{payload['configuration']['api_base']}`", f"- Requests per variant: `{payload['requests']}`", "- Concurrency: `1` (sequential, to avoid provider throttling)", "", "## Before / after", "", "| Metric | Legacy | Compact | Change |", "|---|---:|---:|---:|"]
    for field, label in (("total_ms", "End-to-end"), ("llm_generation_ms", "NVIDIA generation"), ("ttft_ms", "TTFT"), ("retrieval_ms", "Retrieval"), ("persistence_ms", "Persistence"), ("input_tokens_est", "Input tokens"), ("output_tokens", "Output tokens"), ("answer_chars", "Answer chars")):
        lines.append(f"| {label} median | {value(a, field)} | {value(b, field)} | {change(field)} |")
    lines += [f"| Success rate | {a['success_rate']:.1%} | {b['success_rate']:.1%} | {(b['success_rate'] - a['success_rate']):+.1%} |", f"| HTTP 429 rate | {a['rate_429']:.1%} | {b['rate_429']:.1%} | {(b['rate_429'] - a['rate_429']):+.1%} |", "", "## p95", "", f"- Legacy end-to-end p95: **{value(a, 'total_ms', 'p95')}**; compact: **{value(b, 'total_ms', 'p95')}**.", f"- Legacy NVIDIA p95: **{value(a, 'llm_generation_ms', 'p95')}**; compact: **{value(b, 'llm_generation_ms', 'p95')}**.", "", "## Prompt composition", "", f"- Legacy system prompt: **{value(a, 'system_prompt_tokens_est')} tokens**; compact system prompt: **{value(b, 'system_prompt_tokens_est')} tokens**.", f"- Legacy full prompt: **{value(a, 'input_tokens_est')} tokens**; compact full prompt: **{value(b, 'input_tokens_est')} tokens**.", "", "## Quality proxies", "", "| Proxy | Legacy | Compact |", "|---|---:|---:|"]
    for key in ("relevance", "groundedness", "completeness_proxy", "short_query_handled"):
        lines.append(f"| {key} | {a['quality'][key]:.1%} | {b['quality'][key]:.1%} |")
    lines += ["", "## Interpretation", "", "Prompt reduction is retained only if latency improves without a material drop in the quality proxies. `output_tokens` and `reasoning_chars` are provider-reported when NVIDIA includes them in stream usage; otherwise the JSON records null/zero rather than estimating them.", ""]
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> int:
    from backend.app.config import get_settings

    settings = get_settings()
    payload: dict[str, Any] = {
        "metadata": {"timestamp_utc": datetime.now(timezone.utc).isoformat()},
        "configuration": {"model": settings.openai_model, "api_base": settings.openai_api_base, "provider": "NVIDIA NIM"},
        "queries": QUERIES[:args.requests],
        "requests": args.requests,
        "variants": {},
    }
    for profile in ("legacy", "compact"):
        payload["variants"][profile] = await run_profile(profile, args.requests)
        payload["variants"][profile]["aggregate"] = aggregate(payload["variants"][profile])
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_md).write_text(render(payload), encoding="utf-8")
    print(f"wrote {args.output_json} and {args.output_md}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--output-json", default="benchmarks/nvidia_prompt_optimization.json")
    parser.add_argument("--output-md", default="benchmarks/nvidia_prompt_optimization.md")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
