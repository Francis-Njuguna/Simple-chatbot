"""Controlled NVIDIA NIM vs AgentRouter/Claude provider benchmark.

The script freezes the real Amref retrieval result once per question, then sends
the exact same prompt/context to both providers through ``LLMService.stream_answer``.
AgentRouter therefore keeps the production SSE repair transport active.

Typical invocation (PowerShell; credentials stay out of command output)::

    $hit = Select-String .env -Pattern 'nvapi-[A-Za-z0-9_-]+' | Select-Object -First 1
    $env:OPENAI_API_KEY = $hit.Matches[0].Value
    $env:OPENAI_API_BASE = 'https://integrate.api.nvidia.com/v1'
    ./.venv/Scripts/python.exe -u scripts/bench_llm_providers.py

The benchmark intentionally omits ``temperature`` for both providers. Claude
Opus 5 rejects that parameter on AgentRouter, so omission is the only identical
generation setting supported by both endpoints. Production files are not
modified.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
AGENTROUTER_MODEL = "claude-opus-5"

# Covers every requested help-desk family, plus conversational, synonym and typo
# forms. Expected terms are deliberately broad: they are a repeatable quality
# signal, not a claim that lexical overlap is a human correctness judgement.
CASES: list[dict[str, Any]] = [
    {"question": "How do I log in to the LMS?", "kind": "lms", "terms": ["lms", "login"]},
    {"question": "How do I log into Moodle?", "kind": "synonym", "terms": ["lms", "moodle", "login"]},
    {"question": "I cant acces moddle, how do I sine in?", "kind": "typo", "terms": ["lms", "login"]},
    {"question": "How do I reset my student portal password?", "kind": "portal", "terms": ["portal", "password"]},
    {"question": "My Student Portal login is not working. What should I do?", "kind": "login", "terms": ["portal", "login", "password"]},
    {"question": "How do I set up Microsoft Authenticator?", "kind": "mfa", "terms": ["authenticator", "microsoft", "scan"]},
    {"question": "How can I enroll for 2FA on Office 365?", "kind": "synonym", "terms": ["authenticator", "mfa", "office"]},
    {"question": "I changed phones and Microsoft Authenticator is blocking my login.", "kind": "login", "terms": ["authenticator", "mfa", "help"]},
    {"question": "How do I register for and take a VAS exam?", "kind": "vas", "terms": ["vas", "exam"]},
    {"question": "Where can I find the Virtual Assessment System training slides?", "kind": "synonym", "terms": ["vas", "training", "exam"]},
    {"question": "What is SMOWL proctoring and how do I use it?", "kind": "smowl", "terms": ["smowl", "exam", "camera"]},
    {"question": "smwol camera not workng during my exam", "kind": "typo", "terms": ["smowl", "camera", "exam"]},
    {"question": "How do I take an exam with webcam monitoring?", "kind": "synonym", "terms": ["smowl", "camera", "exam"]},
    {"question": "How do I access my university email?", "kind": "email", "terms": ["email", "outlook", "microsoft"]},
    {"question": "Where do I sign in to my corporate Outlook account?", "kind": "synonym", "terms": ["email", "outlook", "login"]},
    {"question": "I cannot log in to my Amref systems. Can you help me?", "kind": "conversational", "terms": ["login", "portal", "lms"]},
    {"question": "Hi, I forgot my password and I am locked out. Please help.", "kind": "conversational", "terms": ["password", "reset", "portal"]},
    {"question": "I can log into Moodle but cannot access my assignments.", "kind": "partial", "terms": ["lms", "assignment", "help"]},
    {"question": "How do I use Microsoft Teams for my classes?", "kind": "teams", "terms": ["teams", "microsoft"]},
    {"question": "Hello, how can I contact the AmIU Help Desk?", "kind": "conversational", "terms": ["help", "desk", "amref"]},
]

_TRIAL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "provider_benchmark_trial", default=None
)
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.I)
_URL = re.compile(r"https?://[^\s)\]>]+", re.I)
_FAILURE_MARKERS = (
    "could not generate an answer:",
    "did not start answering within",
    "the answer was cut off:",
)
_REFUSAL_MARKERS = (
    "isn't covered",
    "is not covered",
    "not covered by",
    "outside the",
    "no articles were retrieved",
)


@dataclass
class FrozenInput:
    index: int
    question: str
    kind: str
    expected_terms: list[str]
    context: str
    images: str
    context_sha256: str
    context_chars: int
    source_ids: list[str]
    source_titles: list[str]
    source_urls: list[str]
    retrieval_ms: float
    embedding_ms: float
    search_rerank_ms: float
    hydration_ms: float
    hydration_status: str
    context_build_ms: float


@dataclass
class Observation:
    phase: str
    provider: str
    model: str
    question_index: int
    question: str
    trial: int
    concurrency: int
    order: int
    ok: bool
    ttft_s: float | None
    llm_total_s: float
    total_answer_s: float
    chars: int
    output_tokens_est: int
    chunks: int
    attempts: int
    retries: int
    http_statuses: list[int]
    error: str
    answer: str
    whitespace_preserved: bool
    markdown_present: bool
    citation_present: bool
    correctness: float
    groundedness: float
    completeness: float
    hallucinated_urls: list[str] = field(default_factory=list)


class ObservedTransport:
    """httpx transport wrapper that attributes attempts/status to one trial."""

    def __init__(self, inner: Any, telemetry: dict[str, dict[str, Any]]) -> None:
        self.inner = inner
        self.telemetry = telemetry

    async def handle_async_request(self, request: Any) -> Any:
        trial_id = _TRIAL_ID.get() or "unattributed"
        row = self.telemetry.setdefault(
            trial_id, {"attempts": 0, "statuses": [], "transport_errors": []}
        )
        row["attempts"] += 1
        try:
            response = await self.inner.handle_async_request(request)
            row["statuses"].append(response.status_code)
            return response
        except Exception as exc:  # noqa: BLE001 - measurement must retain cause
            row["transport_errors"].append(type(exc).__name__)
            raise

    async def aclose(self) -> None:
        await self.inner.aclose()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def grounding_score(answer: str, context: str) -> float:
    stop = {
        "the", "and", "that", "this", "with", "from", "your", "you", "for",
        "are", "can", "will", "have", "has", "not", "but", "then", "into",
        "use", "using", "step", "steps", "help", "amref", "article", "source",
    }
    a = {w.lower() for w in _WORD.findall(answer) if w.lower() not in stop}
    c = {w.lower() for w in _WORD.findall(context) if w.lower() not in stop}
    return len(a & c) / len(a) if a else 0.0


def quality_signals(answer: str, frozen: FrozenInput) -> dict[str, Any]:
    low = answer.lower()
    term_hits = sum(1 for term in frozen.expected_terms if term.lower() in low)
    correctness = term_hits / max(1, len(frozen.expected_terms))
    context_urls = {u.rstrip(".,") for u in _URL.findall(frozen.context)}
    answer_urls = {u.rstrip(".,") for u in _URL.findall(answer)}
    hallucinated = sorted(answer_urls - context_urls - {"https://helpdesk.amref.ac.ke"})
    citation = bool(answer_urls & context_urls) or any(
        title.lower() in low for title in frozen.source_titles if title
    )
    refusal = any(marker in low for marker in _REFUSAL_MARKERS)
    # Partial questions may legitimately say the undocumented portion is absent.
    completeness = 1.0 if len(answer) >= 400 else min(1.0, len(answer) / 400.0)
    if refusal and frozen.kind != "partial":
        completeness *= 0.25
    if hallucinated:
        correctness *= 0.5
    return {
        "correctness": correctness,
        "groundedness": grounding_score(answer, frozen.context),
        "completeness": completeness,
        "citation_present": citation,
        "hallucinated_urls": hallucinated,
    }


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001
        return max(0, round(len(text) / 4))


def secret_is_placeholder(value: str) -> bool:
    low = value.lower()
    return any(x in low for x in ("your-", "-here", "<"))


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


async def freeze_inputs() -> list[FrozenInput]:
    print("warming local retrieval stack...", flush=True)
    from backend.app.rag.embeddings import get_embedding_service

    embedder = get_embedding_service()
    await embedder.embed_query_async("benchmark warmup")

    from backend.app.rag.lexical import get_lexical_index
    from backend.app.rag.reranker import get_reranker
    from backend.app.rag.retriever import get_retriever

    get_lexical_index().rebuild()
    reranker = get_reranker()
    reranker.warmup()
    reranker.score_multi(["warmup query"], ["warmup passage"])
    retriever = get_retriever()

    from backend.app.database.session import async_session_factory

    frozen: list[FrozenInput] = []
    hydration_available = True
    hydration_problem = ""
    for index, case in enumerate(CASES):
        start = time.perf_counter()
        t = time.perf_counter()
        embedding = await retriever.embed_query(case["question"])
        embedding_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        chunks, images, _processed = await retriever.retrieve(
            case["question"], query_embedding=embedding
        )
        search_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if hydration_available:
            try:
                async with async_session_factory() as db:
                    chunks, images = await retriever.hydrate_results(db, chunks, images)
                hydration_status = "postgres"
            except Exception as exc:  # noqa: BLE001 - preserve measurable fallback
                hydration_available = False
                hydration_problem = f"{type(exc).__name__}: {exc}"
                hydration_status = "chroma_metadata_fallback"
                print(
                    "  PostgreSQL hydration unavailable; freezing Chroma result metadata "
                    f"for every provider instead ({hydration_problem})",
                    flush=True,
                )
        else:
            hydration_status = "chroma_metadata_fallback"
        hydration_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        context = retriever.format_context(chunks)
        image_context = retriever.format_images(images)
        context_ms = (time.perf_counter() - t) * 1000
        source_ids = list(dict.fromkeys(c.article_id for c in chunks))
        source_titles = list(dict.fromkeys(c.title for c in chunks))
        source_urls = list(dict.fromkeys(c.url for c in chunks))
        row = FrozenInput(
            index=index,
            question=case["question"],
            kind=case["kind"],
            expected_terms=case["terms"],
            context=context,
            images=image_context,
            context_sha256=hashlib.sha256(context.encode("utf-8")).hexdigest(),
            context_chars=len(context),
            source_ids=source_ids,
            source_titles=source_titles,
            source_urls=source_urls,
            retrieval_ms=(time.perf_counter() - start) * 1000,
            embedding_ms=embedding_ms,
            search_rerank_ms=search_ms,
            hydration_ms=hydration_ms,
            hydration_status=hydration_status,
            context_build_ms=context_ms,
        )
        frozen.append(row)
        print(
            f"  froze {index + 1:02d}/{len(CASES)}  {row.retrieval_ms / 1000:5.2f}s  "
            f"{len(chunks)} chunks  {len(context):5d} chars  {row.question[:48]}",
            flush=True,
        )
    return frozen


def build_provider(name: str, telemetry: dict[str, dict[str, Any]]) -> tuple[Any, Any, dict[str, Any]]:
    import httpx
    from langchain_openai import ChatOpenAI

    from backend.app.config import Settings
    from backend.app.rag.llm import LLMService
    from backend.app.rag.sse_repair import SSERepairTransport

    if name == "nvidia":
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not key or secret_is_placeholder(key):
            raise RuntimeError("NVIDIA credential is missing or a placeholder")
        settings = Settings(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY=key,
            OPENAI_API_BASE=os.environ.get("NVIDIA_BASE_URL", NVIDIA_BASE_URL),
            OPENAI_MODEL=os.environ.get("NVIDIA_MODEL", NVIDIA_MODEL),
        )
        model = settings.openai_model
        base_url = settings.openai_api_base
        default_headers = None
    elif name == "agentrouter":
        settings = Settings(LLM_PROVIDER="agentrouter", AGENTROUTER_MODEL=AGENTROUTER_MODEL)
        key = settings.agentrouter_api_key
        if not key or secret_is_placeholder(key):
            raise RuntimeError("AgentRouter credential is missing or a placeholder")
        model = settings.agentrouter_model
        base_url = settings.agentrouter_base_url
        default_headers = (
            {"User-Agent": settings.agentrouter_user_agent}
            if settings.agentrouter_user_agent
            else None
        )
    else:
        raise ValueError(name)

    timeout = httpx.Timeout(
        connect=settings.llm_connect_timeout,
        read=float(settings.llm_timeout),
        write=float(settings.llm_timeout),
        pool=settings.llm_connect_timeout,
    )
    limits = httpx.Limits(
        max_connections=settings.llm_max_connections,
        max_keepalive_connections=settings.llm_max_keepalive_connections,
    )
    inner = httpx.AsyncHTTPTransport(limits=limits)
    observed = ObservedTransport(inner, telemetry)
    transport: Any = SSERepairTransport(observed) if name == "agentrouter" else observed
    client = httpx.AsyncClient(transport=transport, timeout=timeout)
    kwargs: dict[str, Any] = {}
    if default_headers:
        kwargs["default_headers"] = default_headers
    chat = ChatOpenAI(
        model=model,
        api_key=key,
        base_url=base_url,
        temperature=None,
        max_tokens=settings.llm_max_tokens,
        timeout=timeout,
        max_retries=settings.llm_max_retries,
        stream_chunk_timeout=settings.llm_stream_stall_timeout or None,
        http_async_client=client,
        **kwargs,
    )
    service = LLMService.__new__(LLMService)
    service.settings = settings
    service._llm = chat
    description = {
        "provider": name,
        "model": model,
        "base_url": str(base_url),
        "credential_present": True,
        "temperature": "omitted",
        "max_tokens": settings.llm_max_tokens,
        "connect_timeout_s": settings.llm_connect_timeout,
        "read_timeout_s": settings.llm_timeout,
        "first_token_timeout_s": settings.llm_first_token_timeout,
        "stream_stall_timeout_s": settings.llm_stream_stall_timeout,
        "max_retries": settings.llm_max_retries,
        "sse_repair_active": name == "agentrouter",
    }
    return service, client, description


async def measure(
    provider: str,
    service: Any,
    telemetry: dict[str, dict[str, Any]],
    frozen: FrozenInput,
    *,
    phase: str,
    trial: int,
    concurrency: int,
    order: int,
    unique: str,
) -> Observation:
    trial_id = f"{phase}:{provider}:{unique}"
    token = _TRIAL_ID.set(trial_id)
    started = time.perf_counter()
    first: float | None = None
    parts: list[str] = []
    chunk_count = 0
    error = ""
    try:
        async for text in service.stream_answer(
            question=frozen.question,
            context=frozen.context,
            history="No prior conversation.",
            images=frozen.images,
        ):
            if text:
                if first is None:
                    first = time.perf_counter() - started
                parts.append(text)
                chunk_count += 1
    except Exception as exc:  # stream_answer normally converts errors to text
        error = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        _TRIAL_ID.reset(token)
    llm_total = time.perf_counter() - started
    answer = "".join(parts)
    low = answer.lower()
    if not error:
        marker = next((m for m in _FAILURE_MARKERS if m in low), "")
        if marker:
            error = marker
    wire = telemetry.get(trial_id, {})
    statuses = list(wire.get("statuses", []))
    attempts = int(wire.get("attempts", 0))
    if not error and wire.get("transport_errors"):
        error = ",".join(wire["transport_errors"])
    ok = bool(answer) and not error and (not statuses or statuses[-1] < 400)
    quality = quality_signals(answer, frozen) if ok else {
        "correctness": 0.0,
        "groundedness": 0.0,
        "completeness": 0.0,
        "citation_present": False,
        "hallucinated_urls": [],
    }
    return Observation(
        phase=phase,
        provider=provider,
        model=service._configured_model(),
        question_index=frozen.index,
        question=frozen.question,
        trial=trial,
        concurrency=concurrency,
        order=order,
        ok=ok,
        ttft_s=first,
        llm_total_s=llm_total,
        total_answer_s=llm_total + frozen.retrieval_ms / 1000.0,
        chars=len(answer),
        output_tokens_est=estimate_tokens(answer),
        chunks=chunk_count,
        attempts=attempts,
        retries=max(0, attempts - 1),
        http_statuses=statuses,
        error=error,
        answer=answer,
        whitespace_preserved=any(
            part[:1].isspace() or part[-1:].isspace() for part in parts if part
        ),
        markdown_present=bool(re.search(r"(^|\n)(#{1,6} |[-*] |\d+\. )|\*\*", answer)),
        citation_present=quality["citation_present"],
        correctness=quality["correctness"],
        groundedness=quality["groundedness"],
        completeness=quality["completeness"],
        hallucinated_urls=quality["hallucinated_urls"],
    )


async def warmup_providers(
    providers: dict[str, Any], telemetry: dict[str, dict[str, Any]], frozen: list[FrozenInput], n: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    print(f"warming providers ({n} requests each; excluded from statistics)...", flush=True)
    for i in range(n):
        order = ["nvidia", "agentrouter"] if i % 2 == 0 else ["agentrouter", "nvidia"]
        for pos, provider in enumerate(order):
            obs = await measure(
                provider, providers[provider], telemetry, frozen[i % len(frozen)],
                phase="warmup", trial=i + 1, concurrency=1, order=pos + 1,
                unique=f"{i}:{pos}",
            )
            rows.append(asdict(obs))
            print(
                f"  warmup {i + 1}/{n} {provider:11s} "
                f"ttft={obs.ttft_s if obs.ttft_s is not None else -1:6.2f}s "
                f"total={obs.llm_total_s:6.2f}s {'OK' if obs.ok else 'FAIL ' + obs.error}",
                flush=True,
            )
    return rows


async def sequential_run(
    providers: dict[str, Any], telemetry: dict[str, dict[str, Any]], frozen: list[FrozenInput],
    trials: int, observations: list[Observation], checkpoint: Any,
) -> None:
    pair = 0
    total_pairs = trials * len(frozen)
    for trial in range(1, trials + 1):
        for item in frozen:
            pair += 1
            order = ["nvidia", "agentrouter"] if pair % 2 else ["agentrouter", "nvidia"]
            for pos, provider in enumerate(order):
                obs = await measure(
                    provider, providers[provider], telemetry, item,
                    phase="sequential", trial=trial, concurrency=1, order=pos + 1,
                    unique=f"{trial}:{item.index}:{pos}",
                )
                observations.append(obs)
                print(
                    f"  seq pair {pair:03d}/{total_pairs} t{trial} q{item.index + 1:02d} "
                    f"{provider:11s} ttft={obs.ttft_s if obs.ttft_s is not None else -1:6.2f}s "
                    f"total={obs.llm_total_s:6.2f}s attempts={obs.attempts} "
                    f"{'OK' if obs.ok else 'FAIL ' + obs.error}",
                    flush=True,
                )
            checkpoint()


async def concurrency_run(
    providers: dict[str, Any], telemetry: dict[str, dict[str, Any]], frozen: list[FrozenInput],
    levels: list[int], requests_per_level: int, observations: list[Observation], checkpoint: Any,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    stable = {"nvidia": True, "agentrouter": True}
    for level_index, level in enumerate(levels):
        if level > 10 and not all(stable.values()):
            batches.append({"concurrency": level, "skipped": True, "reason": "<90% success at concurrency 10"})
            print(f"  concurrency {level} skipped: provider instability at 10", flush=True)
            continue
        provider_order = ["nvidia", "agentrouter"] if level_index % 2 == 0 else ["agentrouter", "nvidia"]
        for provider in provider_order:
            sem = asyncio.Semaphore(level)
            batch_start = time.perf_counter()

            async def one(i: int) -> Observation:
                async with sem:
                    item = frozen[i % len(frozen)]
                    return await measure(
                        provider, providers[provider], telemetry, item,
                        phase="concurrency", trial=i + 1, concurrency=level, order=1,
                        unique=f"c{level}:{i}",
                    )

            print(
                f"  load {provider:11s} concurrency={level:2d} requests={requests_per_level}",
                flush=True,
            )
            rows = await asyncio.gather(*(one(i) for i in range(requests_per_level)))
            wall = time.perf_counter() - batch_start
            observations.extend(rows)
            ok = sum(r.ok for r in rows)
            batch = {
                "provider": provider,
                "concurrency": level,
                "requests": len(rows),
                "successes": ok,
                "wall_s": wall,
                "throughput_rps": ok / wall if wall else 0.0,
            }
            batches.append(batch)
            if level == 10:
                stable[provider] = ok / len(rows) >= 0.90
            print(
                f"    {ok}/{len(rows)} success, wall={wall:.2f}s, "
                f"throughput={batch['throughput_rps']:.3f} answers/s",
                flush=True,
            )
            checkpoint()
    return batches


def sign_test_p(differences: list[float]) -> float | None:
    nonzero = [d for d in differences if abs(d) > 1e-9]
    n = len(nonzero)
    if not n:
        return None
    k = min(sum(d > 0 for d in nonzero), sum(d < 0 for d in nonzero))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def bootstrap_median_ci(values: list[float], seed: int = 20260820) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    meds = [statistics.median(rng.choices(values, k=len(values))) for _ in range(3000)]
    return [percentile(meds, 2.5), percentile(meds, 97.5)]


def summarize(observations: list[Observation], batches: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"providers": {}, "concurrency": {}, "paired": {}}
    for provider in ("nvidia", "agentrouter"):
        seq = [r for r in observations if r.provider == provider and r.phase == "sequential"]
        good = [r for r in seq if r.ok]
        quality = [r for r in good]
        summary["providers"][provider] = {
            "requests": len(seq),
            "successes": len(good),
            "failures": len(seq) - len(good),
            "success_rate": len(good) / len(seq) if seq else 0.0,
            "error_rate": (len(seq) - len(good)) / len(seq) if seq else 1.0,
            "errors": dict(Counter(r.error or "unknown" for r in seq if not r.ok)),
            "http_statuses": dict(Counter(s for r in seq for s in r.http_statuses)),
            "retries": sum(r.retries for r in seq),
            "ttft_s": stats([r.ttft_s for r in good if r.ttft_s is not None]),
            "llm_total_s": stats([r.llm_total_s for r in good]),
            "total_answer_s": stats([r.total_answer_s for r in good]),
            "chars": stats([float(r.chars) for r in good]),
            "output_tokens_est": stats([float(r.output_tokens_est) for r in good]),
            "chunks": stats([float(r.chunks) for r in good]),
            "quality": {
                "correctness": statistics.mean(r.correctness for r in quality) if quality else 0.0,
                "groundedness": statistics.mean(r.groundedness for r in quality) if quality else 0.0,
                "citation_rate": statistics.mean(float(r.citation_present) for r in quality) if quality else 0.0,
                "completeness": statistics.mean(r.completeness for r in quality) if quality else 0.0,
                "hallucinated_url_rate": statistics.mean(float(bool(r.hallucinated_urls)) for r in quality) if quality else 0.0,
                "mean_answer_chars": statistics.mean(r.chars for r in quality) if quality else 0.0,
            },
        }
        summary["concurrency"][provider] = {}
        for level in sorted({r.concurrency for r in observations if r.phase == "concurrency" and r.provider == provider}):
            rows = [r for r in observations if r.phase == "concurrency" and r.provider == provider and r.concurrency == level]
            good_rows = [r for r in rows if r.ok]
            batch = next((b for b in batches if b.get("provider") == provider and b.get("concurrency") == level), {})
            summary["concurrency"][provider][str(level)] = {
                "requests": len(rows),
                "successes": len(good_rows),
                "success_rate": len(good_rows) / len(rows) if rows else 0.0,
                "error_rate": (len(rows) - len(good_rows)) / len(rows) if rows else 1.0,
                "throughput_rps": batch.get("throughput_rps"),
                "ttft_s": stats([r.ttft_s for r in good_rows if r.ttft_s is not None]),
                "llm_total_s": stats([r.llm_total_s for r in good_rows]),
                "errors": dict(Counter(r.error or "unknown" for r in rows if not r.ok)),
                "http_statuses": dict(Counter(s for r in rows for s in r.http_statuses)),
            }

    paired: dict[tuple[int, int], dict[str, Observation]] = {}
    for row in observations:
        if row.phase == "sequential" and row.ok:
            paired.setdefault((row.question_index, row.trial), {})[row.provider] = row
    complete = [pair for pair in paired.values() if set(pair) == {"nvidia", "agentrouter"}]
    for metric in ("ttft_s", "llm_total_s"):
        diffs = [
            float(getattr(pair["agentrouter"], metric)) - float(getattr(pair["nvidia"], metric))
            for pair in complete
            if getattr(pair["agentrouter"], metric) is not None and getattr(pair["nvidia"], metric) is not None
        ]
        ratios = [
            float(getattr(pair["agentrouter"], metric)) / float(getattr(pair["nvidia"], metric))
            for pair in complete
            if getattr(pair["agentrouter"], metric) is not None
            and getattr(pair["nvidia"], metric) not in (None, 0)
        ]
        summary["paired"][metric] = {
            "n": len(diffs),
            "agentrouter_minus_nvidia_s": stats(diffs),
            "agentrouter_over_nvidia_ratio": stats(ratios),
            "median_difference_95pct_bootstrap_ci": bootstrap_median_ci(diffs),
            "two_sided_sign_test_p": sign_test_p(diffs),
        }

    n = summary["providers"].get("nvidia", {})
    a = summary["providers"].get("agentrouter", {})
    if n and a and n["ttft_s"]["median"] and a["ttft_s"]["median"]:
        summary["speedup"] = {
            "nvidia_over_agentrouter_ttft": a["ttft_s"]["median"] / n["ttft_s"]["median"],
            "nvidia_over_agentrouter_total": a["llm_total_s"]["median"] / n["llm_total_s"]["median"],
            "agentrouter_over_nvidia_ttft": n["ttft_s"]["median"] / a["ttft_s"]["median"],
            "agentrouter_over_nvidia_total": n["llm_total_s"]["median"] / a["llm_total_s"]["median"],
        }
    return summary


def quality_score(q: dict[str, float]) -> float:
    return 100 * (
        0.35 * q["correctness"]
        + 0.30 * q["groundedness"]
        + 0.20 * q["citation_rate"]
        + 0.15 * q["completeness"]
    ) * (1 - q["hallucinated_url_rate"])


def choose_winners(summary: dict[str, Any]) -> dict[str, Any]:
    ps = summary.get("providers", {})
    if set(ps) != {"nvidia", "agentrouter"} or not all(ps[p]["requests"] for p in ps):
        return {"status": "not_measured", "recommendation": "Provider comparison incomplete."}
    required = ("ttft_s", "llm_total_s")
    if any(
        ps[provider][metric]["median"] is None
        for provider in ps
        for metric in required
    ):
        return {
            "status": "insufficient_successes",
            "recommendation": "At least one provider has no successful measured stream yet.",
        }
    q = {p: quality_score(ps[p]["quality"]) for p in ps}
    latency_raw = {
        p: statistics.mean([ps[p]["ttft_s"]["median"], ps[p]["llm_total_s"]["median"]])
        for p in ps
    }
    fastest = min(latency_raw.values())
    latency = {p: 100 * fastest / latency_raw[p] for p in ps}
    reliability = {p: 100 * ps[p]["success_rate"] for p in ps}
    best_tp = {
        p: max(
            (v.get("throughput_rps") or 0.0 for v in summary["concurrency"].get(p, {}).values()),
            default=0.0,
        )
        for p in ps
    }
    tp_max = max(best_tp.values()) or 1.0
    throughput = {p: 100 * best_tp[p] / tp_max for p in ps}
    overall = {
        p: 0.40 * latency[p] + 0.25 * reliability[p] + 0.25 * q[p] + 0.10 * throughput[p]
        for p in ps
    }
    single = min(ps, key=lambda p: ps[p]["llm_total_s"]["median"] or float("inf"))
    level = "10" if "10" in summary["concurrency"].get("nvidia", {}) and "10" in summary["concurrency"].get("agentrouter", {}) else "5"
    concurrent = min(
        ps,
        key=lambda p: (
            -summary["concurrency"].get(p, {}).get(level, {}).get("success_rate", 0.0),
            summary["concurrency"].get(p, {}).get(level, {}).get("llm_total_s", {}).get("p95") or float("inf"),
        ),
    )
    quality_winner = "tie" if abs(q["nvidia"] - q["agentrouter"]) < 2 else max(q, key=q.get)
    overall_winner = max(overall, key=overall.get)
    return {
        "single_user": single,
        "concurrent": concurrent,
        "quality": quality_winner,
        "overall": overall_winner,
        "score_components": {
            p: {"latency": latency[p], "reliability": reliability[p], "quality": q[p], "throughput": throughput[p], "overall": overall[p]}
            for p in ps
        },
        "recommendation": (
            "Keep NVIDIA as the primary provider; retain configurable AgentRouter fallback."
            if overall_winner == "nvidia"
            else "Switch the primary to AgentRouter Claude Opus 5 while retaining configurable NVIDIA fallback."
        ),
    }


def fnum(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(payload: dict[str, Any]) -> str:
    s = payload.get("summary", {})
    p = s.get("providers", {})
    winners = payload.get("winners", {})
    lines = [
        "# LLM Provider Benchmark",
        "",
        f"Timestamp: {payload['metadata']['timestamp_utc']}",
        f"Git commit: `{payload['metadata']['git_commit']}`",
        f"Python: {payload['metadata']['python_version']}",
        f"Questions: {payload['configuration']['questions']}; trials: {payload['configuration']['trials']}",
        "",
    ]
    if set(p) != {"nvidia", "agentrouter"}:
        lines += ["Benchmark incomplete; see `availability` and `fatal_error` in the JSON report.", ""]
        return "\n".join(lines)
    lines += [
        "| Metric | NVIDIA | AgentRouter |",
        "|---|---:|---:|",
        f"| Model | {payload['providers']['nvidia']['model']} | {payload['providers']['agentrouter']['model']} |",
        f"| Sequential requests | {p['nvidia']['requests']} | {p['agentrouter']['requests']} |",
        f"| TTFT p50 (s) | {fnum(p['nvidia']['ttft_s']['p50'])} | {fnum(p['agentrouter']['ttft_s']['p50'])} |",
        f"| TTFT p95 (s) | {fnum(p['nvidia']['ttft_s']['p95'])} | {fnum(p['agentrouter']['ttft_s']['p95'])} |",
        f"| TTFT p99 (s) | {fnum(p['nvidia']['ttft_s']['p99'])} | {fnum(p['agentrouter']['ttft_s']['p99'])} |",
        f"| LLM total p50 (s) | {fnum(p['nvidia']['llm_total_s']['p50'])} | {fnum(p['agentrouter']['llm_total_s']['p50'])} |",
        f"| LLM total p95 (s) | {fnum(p['nvidia']['llm_total_s']['p95'])} | {fnum(p['agentrouter']['llm_total_s']['p95'])} |",
        f"| LLM total p99 (s) | {fnum(p['nvidia']['llm_total_s']['p99'])} | {fnum(p['agentrouter']['llm_total_s']['p99'])} |",
        f"| Success rate | {p['nvidia']['success_rate']:.1%} | {p['agentrouter']['success_rate']:.1%} |",
        f"| Quality score | {fnum(winners['score_components']['nvidia']['quality'])} | {fnum(winners['score_components']['agentrouter']['quality'])} |",
        f"| Overall score | {fnum(winners['score_components']['nvidia']['overall'])} | {fnum(winners['score_components']['agentrouter']['overall'])} |",
        "",
        "## Concurrency",
        "",
        "| Provider | Users | Success | TTFT p50 | TTFT p95 | Total p50 | Total p95 | Throughput |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in ("nvidia", "agentrouter"):
        for level, row in s.get("concurrency", {}).get(provider, {}).items():
            lines.append(
                f"| {provider} | {level} | {row['success_rate']:.1%} | "
                f"{fnum(row['ttft_s']['p50'])}s | {fnum(row['ttft_s']['p95'])}s | "
                f"{fnum(row['llm_total_s']['p50'])}s | {fnum(row['llm_total_s']['p95'])}s | "
                f"{fnum(row['throughput_rps'], 3)} answers/s |"
            )
    lines += [
        "",
        "## Winner",
        "",
        f"Single-user: **{winners['single_user']}**",
        f"Concurrent: **{winners['concurrent']}**",
        f"Quality: **{winners['quality']}**",
        f"Overall: **{winners['overall']}**",
        "",
        f"Recommendation: {winners['recommendation']}",
        "",
        "Quality metrics are repeatable automated proxies (term coverage, lexical grounding, citations, completeness and URL hallucination), not a substitute for human review of the included raw answers.",
    ]
    return "\n".join(lines) + "\n"


async def async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    observations: list[Observation] = []
    batches: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    telemetry: dict[str, dict[str, Any]] = {}
    providers: dict[str, Any] = {}
    clients: list[Any] = []
    provider_info: dict[str, Any] = {}
    frozen: list[FrozenInput] = []

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "hostname": platform.node(),
    }
    configuration = {
        "questions": len(CASES),
        "trials": args.trials,
        "warmups_per_provider": args.warmups,
        "concurrency_levels": args.concurrency,
        "requests_per_concurrency_level": args.requests_per_level,
        "provider_order": "alternating paired AB/BA",
        "retrieval": "once per question, frozen and shared",
        "history": "No prior conversation.",
        "temperature": "omitted identically for both providers",
    }

    def payload(fatal_error: str = "") -> dict[str, Any]:
        summary = summarize(observations, batches) if observations else {}
        winners = choose_winners(summary) if summary else {"status": "not_measured"}
        return {
            "metadata": metadata,
            "configuration": configuration,
            "providers": provider_info,
            "availability": {name: name in providers for name in ("nvidia", "agentrouter")},
            "fatal_error": fatal_error,
            "frozen_inputs": [asdict(row) for row in frozen],
            "warmups_excluded": warmups,
            "observations": [asdict(row) for row in observations],
            "concurrency_batches": batches,
            "summary": summary,
            "winners": winners,
        }

    def checkpoint(fatal_error: str = "") -> None:
        data = payload(fatal_error)
        output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        output_md.write_text(render_markdown(data), encoding="utf-8")

    try:
        for name in ("nvidia", "agentrouter"):
            service, client, info = build_provider(name, telemetry)
            providers[name] = service
            clients.append(client)
            provider_info[name] = info
        print("providers validated without displaying credentials", flush=True)
        for name, info in provider_info.items():
            print(
                f"  {name:11s} model={info['model']} base={info['base_url']} "
                f"repair={info['sse_repair_active']}",
                flush=True,
            )

        frozen = await freeze_inputs()
        checkpoint()
        warmups.extend(await warmup_providers(providers, telemetry, frozen, args.warmups))
        checkpoint()
        print("running paired sequential benchmark...", flush=True)
        await sequential_run(providers, telemetry, frozen, args.trials, observations, checkpoint)
        if not args.skip_concurrency:
            print("running controlled concurrency benchmark...", flush=True)
            batches.extend(
                await concurrency_run(
                    providers, telemetry, frozen, args.concurrency,
                    args.requests_per_level, observations, checkpoint,
                )
            )
        checkpoint()
        print(f"wrote {output_json} and {output_md}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        print(f"FATAL: {message}", file=sys.stderr, flush=True)
        checkpoint(message)
        return 2
    finally:
        for client in clients:
            await client.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--concurrency", default="1,5,10,20")
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--skip-concurrency", action="store_true")
    parser.add_argument("--output-json", default="bench_llm_providers.json")
    parser.add_argument("--output-md", default="bench_llm_providers.md")
    args = parser.parse_args()
    args.concurrency = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if args.trials < 1 or args.warmups < 3 or len(CASES) < 20:
        parser.error("requires >=1 trial, >=3 warmups, and >=20 questions")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
