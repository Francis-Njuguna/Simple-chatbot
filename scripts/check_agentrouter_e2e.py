"""End-to-end check of the AgentRouter provider through the real RAG service.

Runs the queries that were reported as broken and reports, per query:

* whether the answer declined ("the knowledge base has nothing on this") even
  though chunks were retrieved — the reported bug,
* whether the provider itself failed, which ``generate_answer`` returns as
  ordinary answer text rather than raising,
* whether every citation carries a non-empty title and URL — empty source
  entries were the second reported bug,
* confidence and latency.

Exit codes: 0 clean, 1 a real failure (wrongly declined, or empty citations),
2 provider outage — nothing was measured.

    python scripts/check_agentrouter_e2e.py [--show]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings  # noqa: E402
from backend.app.services.rag_service import RAGService  # noqa: E402

QUERIES = [
    "MFA",
    "how do I use MFA",
    "how do I set up Microsoft Authenticator?",
    "I forgot my MFA",
    "how do I log into Moodle?",
    "student email",
]

# Phrases that mean "the knowledge base does not cover this". Rule 8 tells the
# model to vary its wording, so this is a family of markers, not one string.
DECLINE_MARKERS = (
    "didn't retrieve",
    "did not retrieve",
    "no relevant articles",
    "isn't covered by the amref",
    "is not covered by the amref",
    "not covered in the amref",
    "doesn't appear in the amref",
    "does not appear in the amref",
    "i don't have information",
    "i do not have information",
    "no information about",
)

PROVIDER_MARKERS = (
    "could not generate an answer",
    "not a gap in the knowledge base",
    "check the server logs",
)


def declined(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in DECLINE_MARKERS)


def provider_failed(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in PROVIDER_MARKERS)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="print full answers")
    args = parser.parse_args()

    settings = get_settings()
    print(f"provider : {settings.llm_provider}")
    print(f"model    : {settings.agentrouter_model}")
    print(f"base_url : {settings.agentrouter_base_url}")
    print()

    service = RAGService()
    outages = 0
    wrong_declines: list[str] = []
    empty_citations: list[str] = []

    header = f"{'query':<42} {'src':>4} {'conf':>6} {'s':>6}  verdict"
    print(header)
    print("-" * len(header))

    for query in QUERIES:
        started = time.perf_counter()
        try:
            response = await service.chat(query)
        except Exception as exc:  # noqa: BLE001
            print(f"{query:<42} {'-':>4} {'-':>6} {'-':>6}  RAISED {type(exc).__name__}")
            outages += 1
            continue
        elapsed = time.perf_counter() - started

        answer = response.answer
        sources = response.sources
        titled = [s for s in sources if s.title.strip()]
        linked = [s for s in sources if s.url.strip()]

        if provider_failed(answer):
            verdict = "PROVIDER FAILED"
            outages += 1
        elif declined(answer) and sources:
            verdict = f"WRONGLY DECLINED ({len(sources)} chunks were retrieved)"
            wrong_declines.append(query)
        elif declined(answer):
            verdict = "declined (no chunks retrieved)"
        elif sources and (len(titled) < len(sources) or len(linked) < len(sources)):
            verdict = f"answered, CITATIONS INCOMPLETE ({len(titled)}/{len(sources)} titled)"
            empty_citations.append(query)
        else:
            verdict = "answered"

        print(
            f"{query:<42} {len(sources):>4} {response.confidence:>6.3f} "
            f"{elapsed:>6.1f}  {verdict}"
        )

        if args.show:
            print(f"\n{'=' * 78}\nQ: {query}\n{'-' * 78}\n{answer}")
            if sources:
                print(f"{'-' * 78}\nsources:")
                for s in sources:
                    print(f"  - {s.title or '<EMPTY TITLE>'} → {s.url or '<EMPTY URL>'}")
            print(f"{'=' * 78}\n")

    print()
    if outages:
        print(f"RESULT: provider failed on {outages}/{len(QUERIES)} queries — not measured")
        return 2
    if wrong_declines:
        print(f"RESULT: FAIL — declined despite retrieved context: {wrong_declines}")
        return 1
    if empty_citations:
        print(f"RESULT: FAIL — incomplete citations: {empty_citations}")
        return 1
    print(f"RESULT: PASS — {len(QUERIES)} queries answered, citations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
