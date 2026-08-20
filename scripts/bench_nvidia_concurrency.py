"""NVIDIA-only streaming load test using the completed benchmark's frozen inputs.

This measures provider answer generation, not retrieval. The inputs are loaded
from ``bench_llm_providers.json`` and their contexts are never recomputed.
Each level sends 100 requests, using up to the requested number concurrently.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

from bench_llm_providers import (  # noqa: E402
    FrozenInput,
    Observation,
    build_provider,
    measure,
    stats,
)


def load_frozen(path: Path) -> list[FrozenInput]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frozen_inputs", [])
    if not rows:
        raise RuntimeError(f"no frozen inputs in {path}")
    return [FrozenInput(**row) for row in rows]


def pct(rows: list[Observation], field: str, p: float) -> float | None:
    values = [float(getattr(row, field)) for row in rows if getattr(row, field) is not None]
    return stats(values).get(f"p{int(p)}")


async def run_level(
    service: object,
    frozen: list[FrozenInput],
    telemetry: dict[str, dict],
    level: int,
    requests: int,
    batch_index: int,
) -> tuple[dict, list[Observation]]:
    semaphore = asyncio.Semaphore(level)
    started = time.perf_counter()

    async def one(index: int) -> Observation:
        async with semaphore:
            item = frozen[index % len(frozen)]
            return await measure(
                "nvidia",
                service,
                telemetry,
                item,
                phase="nvidia_concurrency",
                trial=index + 1,
                concurrency=level,
                order=1,
                unique=f"level{level}:batch{batch_index}:request{index}",
            )

    rows = await asyncio.gather(*(one(index) for index in range(requests)))
    wall = time.perf_counter() - started
    good = [row for row in rows if row.ok]
    errors: dict[str, int] = {}
    for row in rows:
        if not row.ok:
            errors[row.error or "unknown"] = errors.get(row.error or "unknown", 0) + 1
    result = {
        "provider": "nvidia",
        "model": service._configured_model(),
        "concurrency": level,
        "requests": requests,
        "successes": len(good),
        "failures": len(rows) - len(good),
        "success_rate": len(good) / len(rows) if rows else 0.0,
        "error_rate": (len(rows) - len(good)) / len(rows) if rows else 1.0,
        "wall_s": wall,
        "throughput_answers_per_s": len(good) / wall if wall else 0.0,
        "ttft_s": stats([row.ttft_s for row in good if row.ttft_s is not None]),
        "llm_generation_s": stats([row.llm_total_s for row in good]),
        "total_answer_s_including_frozen_retrieval": stats(
            [row.total_answer_s for row in good]
        ),
        "output_chars": stats([float(row.chars) for row in good]),
        "output_tokens_est": stats([float(row.output_tokens_est) for row in good]),
        "retries": sum(row.retries for row in rows),
        "errors": errors,
        "http_statuses": {
            str(status): sum(row.http_statuses.count(status) for row in rows)
            for status in sorted({status for row in rows for status in row.http_statuses})
        },
    }
    return result, rows


def markdown(payload: dict) -> str:
    lines = [
        "# NVIDIA Concurrency Benchmark",
        "",
        f"Timestamp: {payload['metadata']['timestamp_utc']}",
        f"Model: `{payload['provider']['model']}`",
        f"Frozen contexts: {payload['configuration']['frozen_contexts']}",
        f"Requests per level: {payload['configuration']['requests_per_level']}",
        "",
        "| Concurrency | Success | TTFT p50 | TTFT p95 | Gen p50 | Gen p95 | Total p50 | Total p95 | Throughput |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["levels"]:
        def f(metric: str, p: int) -> str:
            value = row[metric].get(f"p{p}")
            return "n/a" if value is None else f"{value:.2f}s"

        lines.append(
            f"| {row['concurrency']} | {row['success_rate']:.1%} | "
            f"{f('ttft_s', 50)} | {f('ttft_s', 95)} | "
            f"{f('llm_generation_s', 50)} | {f('llm_generation_s', 95)} | "
            f"{f('total_answer_s_including_frozen_retrieval', 50)} | "
            f"{f('total_answer_s_including_frozen_retrieval', 95)} | "
            f"{row['throughput_answers_per_s']:.3f}/s |"
        )
    lines += [
        "",
        "Generation latency is measured from request initiation until the final streamed text chunk.",
        "Total answer latency adds the frozen context's original local retrieval time once; retrieval was not rerun per provider request.",
    ]
    return "\n".join(lines) + "\n"


async def main(args: argparse.Namespace) -> int:
    frozen = load_frozen(Path(args.frozen))
    telemetry: dict[str, dict] = {}
    service, client, provider = build_provider("nvidia", telemetry)
    try:
        print(
            f"NVIDIA model={provider['model']} endpoint={provider['base_url']} "
            "credential validated without display",
            flush=True,
        )
        print(f"warming with {args.warmups} excluded requests...", flush=True)
        for index in range(args.warmups):
            row = (await run_level(service, frozen, telemetry, 1, 1, index))[1][0]
            print(
                f"  warmup {index + 1}: ttft={row.ttft_s or -1:.2f}s "
                f"generation={row.llm_total_s:.2f}s ok={row.ok}",
                flush=True,
            )

        results: list[dict] = []
        raw_rows: list[dict] = []
        for batch_index, level in enumerate(args.levels):
            print(
                f"running concurrency={level} requests={args.requests_per_level}",
                flush=True,
            )
            result, rows = await run_level(
                service, frozen, telemetry, level, args.requests_per_level, batch_index
            )
            results.append(result)
            raw_rows.extend(asdict(row) for row in rows)
            print(
                f"  success={result['successes']}/{result['requests']} "
                f"wall={result['wall_s']:.2f}s "
                f"throughput={result['throughput_answers_per_s']:.3f}/s "
                f"ttft_p50={result['ttft_s']['p50'] or -1:.2f}s "
                f"generation_p50={result['llm_generation_s']['p50'] or -1:.2f}s",
                flush=True,
            )

        payload = {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpus": __import__("os").cpu_count(),
            },
            "provider": provider,
            "configuration": {
                "frozen_input_file": args.frozen,
                "frozen_contexts": len(frozen),
                "requests_per_level": args.requests_per_level,
                "levels": args.levels,
                "warmups_excluded": args.warmups,
                "retrieval": "frozen from completed paired benchmark; not rerun",
                "streaming": True,
            },
            "levels": results,
            "raw_observations": raw_rows,
        }
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        Path(args.output_md).write_text(markdown(payload), encoding="utf-8")
        print(f"wrote {args.output_json} and {args.output_md}", flush=True)
        return 0
    finally:
        await client.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default="bench_llm_providers.json")
    parser.add_argument("--levels", default="1,5,10,20,50,100")
    parser.add_argument("--requests-per-level", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output-json", default="bench_nvidia_concurrency.json")
    parser.add_argument("--output-md", default="bench_nvidia_concurrency.md")
    args = parser.parse_args()
    args.levels = [int(value) for value in args.levels.split(",") if value.strip()]
    if any(level < 1 for level in args.levels):
        parser.error("concurrency levels must be positive")
    if args.requests_per_level < max(args.levels):
        parser.error("requests per level must be at least the maximum concurrency")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
