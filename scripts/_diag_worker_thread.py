"""Is torch slow inside anyio worker threads? That is where the server runs it.

Both slow stages in the production trace (embedding 629ms for one short query,
rerank 271ms/pair) are dispatched with anyio.to_thread.run_sync. A benchmark
script calls the same code on the main thread and sees ~25ms/pair. If torch
loses its intra-op parallelism in a worker thread, that single fact explains the
whole gap, and no faster model or backend is needed to fix it.

Measures the same warm predict four ways:
  main thread                     - what every bench script measures
  anyio worker thread             - what the server actually does
  anyio worker, threads re-set    - set_num_threads() from inside the worker
  anyio worker, repeated          - does the first worker call pay a one-off?

Run:  ./.venv/Scripts/python.exe -u scripts/_diag_worker_thread.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anyio  # noqa: E402
import torch  # noqa: E402

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag.reranker import get_reranker  # noqa: E402

settings = get_settings()
torch.set_num_threads(settings.torch_num_threads or torch.get_num_threads())

QUERY = "how to download SMOWL"
BASE = (
    "SMOWL is the proctoring tool used for online examinations at AMIU. Before "
    "your first proctored exam you must install the SMOWL extension in the "
    "browser you will use to sit the exam. Log in to the LMS, open the course, "
    "and select the SMOWL registration activity, then allow camera access. "
)
# Production scored shortlist x forms pairs in ONE logical stage. Use the count
# the slow logs actually reflect (16 shortlist x 2 forms) so ms/pair is comparable.
PAIRS = 32
BATCH = [(QUERY, BASE[:400]) for _ in range(PAIRS)]

reranker = get_reranker()
model = reranker._ensure_model()
assert model is not None, "reranker unavailable"


def predict_once() -> float:
    t = time.perf_counter()
    model.predict(BATCH, show_progress_bar=False)
    return (time.perf_counter() - t) * 1000


def report(label: str, times: list[float], threads: int | None = None) -> float:
    best = min(times)
    extra = f"  torch.get_num_threads()={threads}" if threads is not None else ""
    print(
        f"{label:<32}" + " ".join(f"{t:7.0f}" for t in times)
        + f"  | min {best:7.0f}ms  ({best / PAIRS:6.1f}ms/pair){extra}"
    )
    return best


print(f"model         {settings.rerank_model}")
print(f"quantize      {settings.rerank_quantize}")
print(f"main thread   torch.get_num_threads()={torch.get_num_threads()}")
print(f"batch         {PAIRS} pairs (shortlist 16 x 2 forms, as the slow logs)")
print()

# Warm on the main thread.
for _ in range(3):
    predict_once()

main_times = [predict_once() for _ in range(4)]
main_best = report("main thread", main_times, torch.get_num_threads())


async def in_worker(n: int, reset_threads: int | None = None) -> tuple[list[float], int]:
    observed = -1

    def work() -> float:
        nonlocal observed
        if reset_threads is not None:
            torch.set_num_threads(reset_threads)
        observed = torch.get_num_threads()
        return predict_once()

    times = [await anyio.to_thread.run_sync(work) for _ in range(n)]
    return times, observed


async def main() -> None:
    times, obs = await in_worker(4)
    worker_best = report("anyio worker thread", times, obs)

    times, obs = await in_worker(4, reset_threads=settings.torch_num_threads or 4)
    reset_best = report("anyio worker + set_num_threads", times, obs)

    print()
    print(f"worker / main            = {worker_best / main_best:5.2f}x")
    print(f"worker+reset / main      = {reset_best / main_best:5.2f}x")
    print()
    print(f"active python threads: {threading.active_count()}")
    print()
    print("Production logged ~271ms/pair for this batch size. Compare the")
    print("ms/pair column above to see which execution context reproduces it.")


anyio.run(main)
