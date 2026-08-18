"""Why is rerank 5-12s in the server but 0.37s in a script?

Hypothesis: it is not raw CPU speed. Passages vary in length, tokenisation pads
to the longest item in the batch, so nearly every query presents torch with a
(batch, seq_len) shape it has not seen. If oneDNN/MKL re-selects primitives per
novel shape, every query pays a cold-start cost that a repeated-shape benchmark
never sees.

Three cases, same model, same pair count, all warm:
  A  identical batch every iteration      -> shape cache always hits
  B  passage lengths vary per iteration   -> novel shape most iterations
  C  like B but padded to a fixed length  -> shape constant again

If B >> A and C ~= A, the fix is fixed-shape padding, not a faster model.

Run:  ./.venv/Scripts/python.exe -u scripts/_diag_shape_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from backend.app.config import get_settings  # noqa: E402
from backend.app.rag.reranker import get_reranker  # noqa: E402

settings = get_settings()
torch.set_num_threads(settings.torch_num_threads or torch.get_num_threads())

QUERY = "how to download SMOWL"
SHORTLIST = settings.rerank_shortlist
ITERS = 6

BASE = (
    "SMOWL is the proctoring tool used for online examinations at AMIU. Before "
    "your first proctored exam you must install the SMOWL extension in the "
    "browser you will use to sit the exam. Log in to the LMS, open the course, "
    "and select the SMOWL registration activity, then allow camera access when "
    "the browser prompts you so the reference images can be captured. "
)


def fixed_batch() -> list[str]:
    """Every passage the same length, every iteration."""
    return [BASE[:400] for _ in range(SHORTLIST)]


def varying_batch(i: int) -> list[str]:
    """Lengths shift per iteration, as real KB chunks do (200-500 chars)."""
    return [BASE[: 200 + ((i * 37 + j * 23) % 300)] for j in range(SHORTLIST)]


reranker = get_reranker()
model = reranker._ensure_model()
assert model is not None, "reranker unavailable"

print(f"model         {settings.rerank_model}")
print(f"quantize      {settings.rerank_quantize}")
print(f"torch threads {torch.get_num_threads()}")
print(f"batch         {SHORTLIST} pairs, {ITERS} iterations each")
print()

# Warm thoroughly so nothing below pays first-call graph setup.
for _ in range(3):
    model.predict([(QUERY, p) for p in fixed_batch()], show_progress_bar=False)


def run(label: str, batches: list[list[str]], **kw) -> float:
    times = []
    for passages in batches:
        t = time.perf_counter()
        model.predict([(QUERY, p) for p in passages], show_progress_bar=False, **kw)
        times.append((time.perf_counter() - t) * 1000)
    per_pair = [t / SHORTLIST for t in times]
    print(f"{label:<34}", " ".join(f"{t:6.0f}" for t in times), f"  | min {min(times):6.0f}ms  median-ish {sorted(times)[len(times)//2]:6.0f}ms  ({min(per_pair):5.1f}ms/pair best)")
    return min(times)


a = run("A fixed shape", [fixed_batch() for _ in range(ITERS)])
b = run("B varying shape (production-like)", [varying_batch(i) for i in range(ITERS)])
c = run(
    "C varying, padded to max_length",
    [varying_batch(i) for i in range(ITERS)],
    processing_kwargs={"text": {"max_length": 256, "padding": "max_length", "truncation": True}},
)

print()
print(f"B / A  = {b / a:5.2f}x   (>>1 means novel shapes are the problem)")
print(f"C / A  = {c / a:5.2f}x   (~1 means fixed-shape padding fixes it)")
print(f"C / B  = {c / b:5.2f}x   (<1 means padding is a net win)")
