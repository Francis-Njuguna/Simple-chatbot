"""Compare candidate NVIDIA NIM chat models against the recorded baseline.

Why this script exists
----------------------
``OPENAI_MODEL`` was pointing at ``meta/llama-3.1-8b-instruct``, which NVIDIA
retired on 2026-08-26 and now answers with ``410 Gone``. A replacement has to be
chosen on measured evidence, not on model card prose.

Comparability contract
----------------------
The point of this script is that its numbers can be placed in the same table as
the ones already recorded in ``bench_llm_providers.md``. That requires four
things to be *identical* to the baseline run, not merely similar:

1. **The same 20 frozen retrieval contexts**, reloaded from
   ``bench_llm_providers.json`` and never recomputed. Their ``context_sha256``
   values are re-verified on load, so a silently re-embedded or re-ranked
   context cannot masquerade as the original.
2. **The same generation settings** — ``max_tokens`` from
   ``settings.llm_max_tokens`` and ``temperature`` omitted, exactly as
   ``build_provider`` does for the baseline.
3. **The same measurement function** — ``measure()`` is imported, not
   reimplemented, so time-to-first-token and success are defined identically.
4. **The same quality scoring** — ``quality_signals`` / ``quality_score`` are
   imported from the baseline harness.

The old baseline is therefore quoted from JSON rather than re-run: it *cannot*
be re-run, and pretending otherwise would silently drop it from the comparison.

Volume discipline
-----------------
``bench_nvidia_concurrency.md`` establishes that NVIDIA's free tier binds on
**total request volume**, not concurrency: a 60-request run saw zero 429s while
a 600-request run saw 826. This script therefore runs strictly sequentially,
paces requests, counts 429s as a first-class outcome, and stops early rather
than producing a quota curve mislabelled as a latency result.

TTFT on reasoning models
------------------------
``measure()`` starts the TTFT clock at request initiation and stops it at the
first chunk of *answer* text. Nemotron reasoning models emit a separate
``reasoning_content`` field first, which ``_extract_text`` does not yield. So
TTFT here is genuinely "time until a student sees a word", which is the metric
that matters, and it is directly comparable to the baseline's TTFT.

Invocation::

    ./.venv/Scripts/python.exe -u scripts/bench_nemotron_models.py --trials 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from bench_llm_providers import (  # noqa: E402
    FrozenInput,
    Observation,
    build_provider,
    measure,
    quality_score,
    stats,
)

# Candidates confirmed to answer /v1/chat/completions on this account.
# `meta/llama-3.1-8b-instruct` is deliberately absent: it returns 410 Gone.
CANDIDATES = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
]

BASELINE_JSON = Path("bench_llm_providers.json")


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_frozen(path: Path) -> list[FrozenInput]:
    """Reload the baseline's frozen contexts and prove they are unchanged."""
    import hashlib

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frozen_inputs", [])
    if not rows:
        raise RuntimeError(f"no frozen_inputs in {path}")
    frozen = [FrozenInput(**row) for row in rows]
    for item in frozen:
        actual = hashlib.sha256(item.context.encode("utf-8")).hexdigest()
        if actual != item.context_sha256:
            raise RuntimeError(
                f"frozen context {item.index} no longer matches its recorded "
                f"sha256 — comparison to the baseline would be invalid"
            )
    return frozen


def baseline_row(path: Path) -> dict[str, Any]:
    """The recorded llama-3.1-8b numbers. Quoted, because it cannot be re-run."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    nvidia = payload["summary"]["providers"]["nvidia"]
    return {
        "model": payload["providers"]["nvidia"]["model"],
        "status": "retired_410_gone",
        "measured_at": payload["metadata"]["timestamp_utc"],
        "rerunnable": False,
        "requests": nvidia["requests"],
        "success_rate": nvidia["success_rate"],
        "ttft_s": nvidia["ttft_s"],
        "llm_total_s": nvidia["llm_total_s"],
        "quality": nvidia["quality"],
        "quality_score": quality_score(nvidia["quality"]),
        "output_tokens_est": nvidia["output_tokens_est"],
    }


def summarize_model(rows: list[Observation]) -> dict[str, Any]:
    good = [r for r in rows if r.ok]
    quota = sum(1 for r in rows if 429 in (r.http_statuses or []))
    errors: dict[str, int] = {}
    for r in rows:
        if not r.ok:
            errors[r.error or "unknown"] = errors.get(r.error or "unknown", 0) + 1
    quality = {
        "correctness": statistics.mean(r.correctness for r in good) if good else 0.0,
        "groundedness": statistics.mean(r.groundedness for r in good) if good else 0.0,
        "citation_rate": statistics.mean(float(r.citation_present) for r in good) if good else 0.0,
        "completeness": statistics.mean(r.completeness for r in good) if good else 0.0,
        "hallucinated_url_rate": statistics.mean(
            float(bool(r.hallucinated_urls)) for r in good
        ) if good else 0.0,
        "mean_answer_chars": statistics.mean(r.chars for r in good) if good else 0.0,
    }
    return {
        "requests": len(rows),
        "successes": len(good),
        "failures": len(rows) - len(good),
        "success_rate": len(good) / len(rows) if rows else 0.0,
        "quota_429_requests": quota,
        "errors": errors,
        "ttft_s": stats([r.ttft_s for r in good if r.ttft_s is not None]),
        "llm_total_s": stats([r.llm_total_s for r in good]),
        "output_tokens_est": stats([float(r.output_tokens_est) for r in good]),
        "quality": quality,
        "quality_score": quality_score(quality),
        "rerunnable": True,
        "status": "measured",
    }


def fnum(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(payload: dict[str, Any]) -> str:
    models = payload["models"]
    base = payload["baseline"]
    cfg = payload["configuration"]
    first_token_budget = cfg["first_token_timeout_s"]

    lines = [
        "# Nemotron Model Comparison (NVIDIA NIM)",
        "",
        f"Timestamp: {payload['metadata']['timestamp_utc']}",
        f"Git commit: `{payload['metadata']['git_commit']}`",
        f"Python: {payload['metadata']['python_version'].splitlines()[0]}",
        "",
        (
            f"Frozen contexts: {cfg['frozen_contexts']} (sha256-verified identical to "
            f"the baseline run); trials: {cfg['trials']}; "
            f"requests per model: {cfg['requests_per_model']}"
        ),
        (
            "Generation settings held identical to baseline: "
            f"max_tokens={cfg['max_tokens']}, temperature={cfg['temperature']}, "
            f"first-token budget={first_token_budget}s"
        ),
        "",
        "## Why the old model is quoted, not re-measured",
        "",
        (
            f"`{base['model']}` was the configured model and the baseline in "
            "`bench_llm_providers.md`. NVIDIA retired it on 2026-08-26 and the endpoint "
            "now returns `410 Gone`, so it can never be re-run. Its row below is quoted "
            f"verbatim from `{BASELINE_JSON.name}` (measured {base['measured_at']}) and "
            "is comparable only because this run reuses the same frozen contexts, the "
            "same generation settings and the same measurement code."
        ),
        "",
        "## Results",
        "",
        "| Model | Status | Reqs | Success | TTFT p50 | TTFT p95 | Total p50 | Total p95 | Out tok p50 | Quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    rows = [(base["model"], base, "quoted (410 Gone)")] + [
        (name, models[name], "measured") for name in models
    ]
    for name, row, status in rows:
        lines.append(
            f"| `{name}` | {status} | {row['requests']} | "
            f"{row['success_rate']:.1%} | "
            f"{fnum(row['ttft_s'].get('p50'))}s | {fnum(row['ttft_s'].get('p95'))}s | "
            f"{fnum(row['llm_total_s'].get('p50'))}s | {fnum(row['llm_total_s'].get('p95'))}s | "
            f"{fnum(row['output_tokens_est'].get('p50'), 0)} | "
            f"{fnum(row['quality_score'])} |"
        )

    lines += [
        "",
        "## Quality breakdown",
        "",
        "| Model | Correctness | Groundedness | Citation rate | Completeness | Hallucinated URL rate | Mean chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row, _ in rows:
        q = row["quality"]
        lines.append(
            f"| `{name}` | {q['correctness']:.3f} | {q['groundedness']:.3f} | "
            f"{q['citation_rate']:.3f} | {q['completeness']:.3f} | "
            f"{q['hallucinated_url_rate']:.3f} | {q['mean_answer_chars']:.0f} |"
        )

    lines += [
        "",
        (
            "Quality is the same weighted proxy the baseline used "
            "(0.35 correctness + 0.30 groundedness + 0.20 citation + 0.15 completeness, "
            "scaled by 1 - hallucinated-URL rate). It is a repeatable lexical signal, "
            "not human review."
        ),
        "",
        "## Failures and quota",
        "",
        "| Model | Failures | 429 requests | Error breakdown |",
        "|---|---:|---:|---|",
    ]
    for name in models:
        row = models[name]
        errs = ", ".join(f"{k} x{v}" for k, v in row["errors"].items()) or "none"
        lines.append(
            f"| `{name}` | {row['failures']} | {row['quota_429_requests']} | {errs} |"
        )

    lines += [
        "",
        (
            "Total 429 responses across the whole run: "
            f"**{payload['totals']['quota_429_requests']}**. Per "
            "`bench_nvidia_concurrency.md`, NVIDIA's free tier binds on total request "
            "volume, so this figure is the check on whether the latency numbers above "
            "are trustworthy or quota-distorted."
        ),
        "",
        "## Scope",
        "",
        (
            "- Sequential only (concurrency 1), which is what production runs: "
            "`LLM_MAX_CONCURRENCY=1`."
        ),
        (
            "- Concurrency scaling is deliberately **not** re-measured. "
            "`bench_nvidia_concurrency.md` shows the free-tier limiter binds before app "
            "capacity does, so such a sweep charts the quota policy, not this server."
        ),
        (
            "- Retrieval is excluded: contexts are frozen, so these are "
            "generation-only figures."
        ),
    ]
    return "\n".join(lines) + "\n"


async def async_main(args: argparse.Namespace) -> int:
    # Credential is taken from the app's own settings and never printed.
    from backend.app.config import get_settings

    settings = get_settings()
    key = (settings.openai_api_key or "").strip()
    if not key:
        print("FATAL: no NVIDIA credential available via settings", file=sys.stderr)
        return 2
    os.environ["NVIDIA_API_KEY"] = key
    os.environ["NVIDIA_BASE_URL"] = settings.openai_api_base or ""

    frozen = load_frozen(BASELINE_JSON)
    print(
        f"loaded {len(frozen)} frozen contexts; all sha256 values verified unchanged",
        flush=True,
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    telemetry: dict[str, dict[str, Any]] = {}
    services: dict[str, Any] = {}
    clients: list[Any] = []

    observations: dict[str, list[Observation]] = {m: [] for m in models}
    raw: list[dict[str, Any]] = []
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    configuration = {
        "frozen_contexts": len(frozen),
        "frozen_source": str(BASELINE_JSON),
        "trials": args.trials,
        "requests_per_model": args.trials * len(frozen),
        "warmups_per_model": args.warmups,
        "concurrency": 1,
        "pace_seconds": args.pace,
        "max_tokens": settings.llm_max_tokens,
        "temperature": "omitted (identical to baseline)",
        "first_token_timeout_s": settings.llm_first_token_timeout,
        "read_timeout_s": settings.llm_timeout,
        "ordering": "round-robin across models per question, rotated per trial",
    }

    def build_payload(fatal: str = "") -> dict[str, Any]:
        summaries = {m: summarize_model(observations[m]) for m in models if observations[m]}
        total_429 = sum(s["quota_429_requests"] for s in summaries.values())
        return {
            "metadata": metadata,
            "configuration": configuration,
            "fatal_error": fatal,
            "baseline": baseline_row(BASELINE_JSON),
            "models": summaries,
            "totals": {"quota_429_requests": total_429},
            "observations": raw,
        }

    def checkpoint(fatal: str = "") -> None:
        payload = build_payload(fatal)
        output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if payload["models"]:
            output_md.write_text(render_markdown(payload), encoding="utf-8")

    try:
        for name in models:
            os.environ["NVIDIA_MODEL"] = name
            service, client, _info = build_provider("nvidia", telemetry)
            services[name] = service
            clients.append(client)
        print(f"built {len(services)} clients (credentials not displayed)", flush=True)

        # Warmup — excluded from statistics. Opens the connection pool and pays
        # any provider-side cold start once per model.
        for name in models:
            for i in range(args.warmups):
                obs = await measure(
                    "nvidia", services[name], telemetry, frozen[i % len(frozen)],
                    phase="warmup", trial=i + 1, concurrency=1, order=1,
                    unique=f"warm:{name}:{i}",
                )
                print(
                    f"  warmup {name:38s} ttft="
                    f"{obs.ttft_s if obs.ttft_s is not None else -1:6.2f}s "
                    f"total={obs.llm_total_s:6.2f}s "
                    f"{'OK' if obs.ok else 'FAIL ' + obs.error[:60]}",
                    flush=True,
                )
                await asyncio.sleep(args.pace)

        aborted = ""
        done = 0
        total = args.trials * len(frozen) * len(models)
        for trial in range(1, args.trials + 1):
            for item in frozen:
                # Rotate model order so provider-side drift cannot systematically
                # favour whichever model always went first.
                order = models[(trial + item.index) % len(models) :] + models[
                    : (trial + item.index) % len(models)
                ]
                for name in order:
                    obs = await measure(
                        "nvidia", services[name], telemetry, item,
                        phase="sequential", trial=trial, concurrency=1, order=1,
                        unique=f"seq:{name}:{trial}:{item.index}",
                    )
                    observations[name].append(obs)
                    raw.append(asdict(obs))
                    done += 1
                    print(
                        f"  [{done:3d}/{total}] t{trial} q{item.index + 1:02d} "
                        f"{name:38s} ttft="
                        f"{obs.ttft_s if obs.ttft_s is not None else -1:6.2f}s "
                        f"total={obs.llm_total_s:6.2f}s "
                        f"tok={obs.output_tokens_est:4d} "
                        f"{'OK' if obs.ok else 'FAIL ' + obs.error[:50]}",
                        flush=True,
                    )
                    await asyncio.sleep(args.pace)

                checkpoint()

                # Volume guard: if the free tier starts refusing, latency numbers
                # stop being latency numbers. Stop and say so.
                recent_429 = sum(
                    1
                    for rows in observations.values()
                    for r in rows[-6:]
                    if 429 in (r.http_statuses or [])
                )
                if recent_429 >= args.quota_abort:
                    aborted = (
                        f"aborted after {done} requests: {recent_429} recent 429s — "
                        "provider quota was binding, further numbers would measure "
                        "the rate limiter rather than the models"
                    )
                    print(f"ABORT: {aborted}", flush=True)
                    break
            if aborted:
                break

        checkpoint(aborted)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(CANDIDATES))
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--pace", type=float, default=0.4, help="seconds between requests")
    p.add_argument("--quota-abort", type=int, default=4)
    p.add_argument("--output-json", default="bench_nemotron_models.json")
    p.add_argument("--output-md", default="bench_nemotron_models.md")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
