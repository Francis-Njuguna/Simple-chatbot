"""Explain the difference between direct and full-RAG NVIDIA generation time."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

QUESTIONS = [
    "How do I set up Microsoft Authenticator?",
    "How do I log in to the LMS?",
    "How do I reset my student portal password?",
    "What is SMOWL proctoring?",
    "Please help me log in.",
]


async def raw_stream(client, settings, key: str, messages: list[dict], temperature) -> dict:
    started = time.perf_counter()
    headers_at = None
    first_visible = None
    reasoning_chars = 0
    answer_chars = 0
    output_tokens = None
    statuses: list[int] = []
    error = ""
    payload = {
        "model": settings.openai_model,
        "messages": messages,
        "max_tokens": settings.llm_max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    try:
        async with client.stream(
            "POST", f"{settings.openai_api_base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=payload,
        ) as response:
            headers_at = time.perf_counter() - started
            statuses.append(response.status_code)
            if response.status_code >= 400:
                error = f"HTTP {response.status_code}"
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
                    if event.get("usage"):
                        output_tokens = event["usage"].get("completion_tokens") or event["usage"].get("output_tokens")
                    for choice in event.get("choices", []) or []:
                        delta = choice.get("delta") or {}
                        reasoning_chars += len(delta.get("reasoning_content") or "")
                        content = delta.get("content") or ""
                        if content:
                            answer_chars += len(content)
                            if first_visible is None:
                                first_visible = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - started
    return {
        "path": "raw_nvidia_sse",
        "headers_s": headers_at,
        "ttft_from_request_s": first_visible,
        "total_s": total,
        "reasoning_chars": reasoning_chars,
        "answer_chars": answer_chars,
        "output_tokens": output_tokens,
        "statuses": statuses,
        "error": error,
    }


async def app_stream(llm, question: str, context: str, images: str) -> dict:
    started = time.perf_counter()
    first_visible = None
    answer_chars = 0
    parts = []
    async for text in llm.stream_answer(question, context, "No prior conversation.", images):
        if text:
            if first_visible is None:
                first_visible = time.perf_counter() - started
            parts.append(text)
            answer_chars += len(text)
    return {
        "path": "LLMService_stream_answer",
        "headers_s": None,
        "ttft_from_request_s": first_visible,
        "total_s": time.perf_counter() - started,
        "reasoning_chars": None,
        "answer_chars": answer_chars,
        "output_tokens": None,
        "statuses": None,
        "error": "" if answer_chars and "could not generate an answer:" not in "".join(parts).lower() else "application error/fallback",
    }


async def main() -> int:
    import httpx
    from backend.app.config import get_settings
    from backend.app.database.session import db_scope
    from backend.app.rag.llm import get_llm_service
    from backend.app.rag.retriever import get_retriever

    settings = get_settings()
    key = (settings.openai_api_key or "").strip()
    llm = get_llm_service()
    retriever = get_retriever()
    await retriever.embed_query("diagnostic warmup")
    from backend.app.rag.reranker import get_reranker
    get_reranker().warmup()
    timeout = httpx.Timeout(connect=settings.llm_connect_timeout, read=90.0, write=90.0, pool=30.0)
    rows = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for question in QUESTIONS:
            embedding = await retriever.embed_query(question)
            chunks, images, _ = await retriever.retrieve(question, query_embedding=embedding)
            async with db_scope("gap_diagnostic_hydration") as db:
                chunks, images = await retriever.hydrate_results(db, chunks, images)
            context = retriever.format_context(chunks)
            image_context = retriever.format_images(images)
            prompt_messages = [
                {"role": "system", "content": str(llm._build_messages(question, context, "No prior conversation.", image_context)[0].content)},
                {"role": "user", "content": str(llm._build_messages(question, context, "No prior conversation.", image_context)[1].content)},
            ]
            temperature = getattr(llm._llm, "temperature", None)
            raw = await raw_stream(client, settings, key, prompt_messages, temperature)
            app = await app_stream(llm, question, context, image_context)
            rows.append({
                "question": question,
                "context_chars": len(context),
                "system_chars": len(prompt_messages[0]["content"]),
                "user_chars": len(prompt_messages[1]["content"]),
                "raw": raw,
                "app": app,
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    Path("benchmarks/nvidia_latency_gap.json").write_text(json.dumps({"model": settings.openai_model, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
