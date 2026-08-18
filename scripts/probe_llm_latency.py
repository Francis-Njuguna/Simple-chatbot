"""Measure time-to-first-*visible*-token per model, through the real stream path.

Why this exists
---------------
The retry/timeout fix bounds the failure path and saves nothing on a healthy
request. What is left on the healthy path is the model, and the log only gives
four usable samples (``llm_first_token`` 4.7s / 11.6s / 15.9s, plus one 90.9s
cascade) — all for ``claude-opus-5``. This probe produces the comparison the log
cannot.

What it measures, and why that framing matters
----------------------------------------------
Not "time to first chunk". ``claude-opus-5`` is a reasoning model: it streams
``reasoning_content`` deltas that carry no answer text, and ``stream_answer``
filters those out (correctly — they are not the answer). So the first chunk off
the wire can arrive seconds before the first chunk a student can *read*. This
measures the second one, by consuming ``stream_answer`` itself rather than
re-implementing it.

Method notes
------------
* Models are run **round-robin**, not model-by-model. This box demonstrably
  stalls whole processes for seconds at a time (see RERANK_LATENCY.md), and
  grouping all of one model's trials together would hand a stall entirely to
  whichever model happened to be running.
* Every trial is printed, never just the mean. With n=3 on a noisy box the
  spread is the finding; a mean would hide it.
* The prompt is a fixed synthetic context sized to the ~4,950 characters
  observed on a real request, so runs are comparable across invocations and
  across models.

Usage::

    ./.venv/Scripts/python.exe -u scripts/probe_llm_latency.py
    ./.venv/Scripts/python.exe -u scripts/probe_llm_latency.py --trials 5 \
        --models claude-haiku-4-5 claude-opus-5

A target may be given as ``provider:model`` to compare across providers, not only
across models on one gateway. A bare name still means agentrouter::

    ./.venv/Scripts/python.exe -u scripts/probe_llm_latency.py \
        --models claude-opus-5 gemini:gemini-2.0-flash

``openai`` means "any OpenAI-compatible endpoint" — NVIDIA NIM, xAI, a local
Qwen — so it needs a base URL and a key. Those come from the environment rather
than from flags, which keeps the key out of shell history and out of this file,
and leaves the running server's ``.env`` untouched::

    OPENAI_API_BASE=https://integrate.api.nvidia.com/v1 OPENAI_API_KEY=nvapi-... \
        ./.venv/Scripts/python.exe -u scripts/probe_llm_latency.py \
        --models openai:meta/llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import Settings  # noqa: E402
from backend.app.rag.llm import LLMService  # noqa: E402

QUESTION = "How do I reset my student portal password?"

# Sized to the ~4,950-char prompt observed on a real request so the input token
# count is representative; the wording is generic on purpose, since what is being
# compared is per-model latency, not retrieval quality.
_ARTICLE = """
Resetting your AMIU student portal password

1. Open the student portal sign-in page and choose "Forgot password".
2. Enter the university email address on your admission letter. Personal
   addresses will not be recognised by the system.
3. A reset link is sent to that address. The link expires after 30 minutes.
4. Choose a new password of at least 10 characters, including one number and
   one symbol. The previous three passwords cannot be reused.
5. Sign in again with the new password. If multi-factor authentication is
   enabled on your account you will also be asked for the six-digit code from
   your authenticator application.

If the reset email does not arrive, check the junk folder first, then confirm
with the registry that the address on file is current. Accounts that have been
dormant for more than one semester are locked and cannot be reset online; these
must be reopened by the ICT help desk in person.
""".strip()

CONTEXT = "\n\n".join(f"[Article {i + 1}]\n{_ARTICLE}" for i in range(4))


# Which Settings field carries the model name, per provider. Everything else a
# provider needs — key, base URL — already comes from the environment, so this
# mapping is the whole of the provider switch.
_MODEL_FIELD = {
    "agentrouter": "AGENTROUTER_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
    "gemini": "GEMINI_MODEL",
    "ollama": "OLLAMA_MODEL",
    "openai": "OPENAI_MODEL",
}


def _split_target(spec: str) -> tuple[str, str]:
    """``"gemini:gemini-2.0-flash"`` -> ``("gemini", "gemini-2.0-flash")``.

    A bare name means agentrouter, so every earlier invocation of this probe
    keeps working. Splits on the *first* colon only: ``ollama:qwen3:4b`` is one
    provider and one tag.
    """
    provider, sep, model = spec.partition(":")
    if not sep:
        return "agentrouter", provider
    if provider not in _MODEL_FIELD:
        raise SystemExit(
            f"unknown provider {provider!r} in {spec!r}; expected one of "
            f"{', '.join(sorted(_MODEL_FIELD))}"
        )
    if not model:
        raise SystemExit(f"no model given in {spec!r}")
    return provider, model


def _service(spec: str) -> LLMService:
    """An LLMService pinned to one provider+model, built without get_settings().

    ``get_settings()`` is ``lru_cache``d on the real environment, so overriding a
    field means constructing ``Settings`` directly and skipping ``__init__``.
    Real environment variables outrank ``.env`` in pydantic-settings, so an
    endpoint can be pointed somewhere else for one invocation without editing
    ``.env`` and without disturbing the running server.
    """
    provider, model = _split_target(spec)
    svc = LLMService.__new__(LLMService)
    svc.settings = Settings(LLM_PROVIDER=provider, **{_MODEL_FIELD[provider]: model})
    svc._llm = svc._build_llm()
    return svc


async def _trial(svc: LLMService) -> tuple[float | None, float, int, str]:
    """Return (seconds to first visible text, total seconds, characters, problem).

    ``problem`` is non-empty when the "answer" was really a status message, and
    it exists because ``stream_answer`` never raises — it catches everything and
    yields prose. A 403 for an unavailable model therefore arrives as a fast,
    clean-looking single chunk, and an earlier version of this probe duly
    reported it as the winning model at 0.99s with a perfect 3/3. Any timing
    conclusion drawn from a table that cannot tell an answer from an error is
    worthless, so the check is part of the measurement, not a nicety.
    """
    started = time.monotonic()
    first: float | None = None
    parts: list[str] = []
    async for text in svc.stream_answer(QUESTION, CONTEXT):
        if first is None:
            first = time.monotonic() - started
        parts.append(text)
    # Accumulated in full rather than sampled: the mid-stream "cut off" notice is
    # appended as the *last* chunk, so checking only the opening would miss it.
    # Answers are capped at LLM_MAX_TOKENS, so this is a few KB at most.
    answer = "".join(parts)

    problem = ""
    if "could not generate an answer:" in answer:
        # Keep the specific clause _error_message chose — "rejected the
        # configured credentials", "does not recognise the configured model" —
        # since that is the actual result of this trial.
        problem = answer.split("could not generate an answer:", 1)[1]
        problem = problem.split(".")[0].strip()[:70] or "provider error"
    elif "did not start answering" in answer:
        problem = "hit the first-token ceiling"
    elif "cut off" in answer:
        problem = "stalled mid-answer"

    return first, time.monotonic() - started, len(answer), problem


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="[PROVIDER:]MODEL",
        default=["claude-opus-5", "claude-haiku-4-5", "claude-sonnet-4-5"],
    )
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print(f"prompt ~{len(CONTEXT):,} chars of context + question")
    print(f"{len(args.models)} targets x {args.trials} trials, round-robin\n")

    # Built one at a time and tolerantly: _build_llm() raises when a provider's
    # key is missing, and one unconfigured target must not cancel the trials for
    # the others — comparing whatever *is* reachable is the point of the run.
    services: dict[str, LLMService] = {}
    unbuildable: dict[str, str] = {}
    for spec in args.models:
        try:
            services[spec] = _service(spec)
        except SystemExit:
            raise  # a malformed spec is a usage error, not a provider outage
        except Exception as exc:  # noqa: BLE001 - report, then try the next one
            unbuildable[spec] = f"{type(exc).__name__}: {exc}"[:90]
            print(f"  skipping {spec} — {unbuildable[spec]}")
    if not services:
        print("\nno target could be built; nothing to measure")
        return 1
    if unbuildable:
        print()

    order = list(services)
    results: dict[str, list[tuple[float | None, float, int, str]]] = {
        m: [] for m in order
    }

    try:
        for trial in range(args.trials):
            for model in order:  # round-robin: see module docstring
                first, total, chars, problem = await _trial(services[model])
                results[model].append((first, total, chars, problem))
                shown = f"{first:6.2f}s" if first is not None else "  none"
                flag = f"  <- NOT AN ANSWER: {problem}" if problem else ""
                print(
                    f"  trial {trial + 1}  {model:34s} "
                    f"first_text {shown}  total {total:6.2f}s  "
                    f"{chars:5d} chars{flag}"
                )
            print()

        print(
            f"{'target':34s} {'first_text (median)':>20s} "
            f"{'total (median)':>16s} {'chars':>7s}  usable"
        )
        for model, rows in results.items():
            # Only real answers. A failed trial would otherwise contribute a
            # flattering sub-second time for a request that produced no answer.
            good = [(f, t, c) for f, t, c, p in rows if f is not None and not p]
            if not good:
                reasons = {p for *_, p in rows if p}
                print(f"{model:34s} {'no usable answer':>20s}  ({'; '.join(reasons)})")
                continue
            print(
                f"{model:34s} {statistics.median(f for f, _, _ in good):19.2f}s "
                f"{statistics.median(t for _, t, _ in good):15.2f}s "
                f"{statistics.median(c for _, _, c in good):7.0f}  "
                f"{len(good)}/{len(rows)}"
            )
        # Repeated under the table so a skipped target is not mistaken for one
        # that was never asked for, when only the summary gets pasted somewhere.
        for spec, reason in unbuildable.items():
            print(f"{spec:34s} {'not built':>20s}  ({reason})")
    finally:
        # These clients hold real connection pools; a probe should not leak them.
        for svc in services.values():
            client = getattr(svc._llm, "root_async_client", None)
            if client is not None:
                await client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
