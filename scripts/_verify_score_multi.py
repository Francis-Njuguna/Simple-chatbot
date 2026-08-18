"""Verify score_multi() == per-form score() + max-pool, and measure the gain.

The batching change in reranker.py claims to be numerically identical to the
sequential loop it replaced. "Claims" is not good enough for the stage that
decides whether an off-topic question gets declined, so this asserts it on real
KB passages at the shipped shortlist/form counts, then times both paths.

Run:  ./.venv/Scripts/python.exe -u scripts/_verify_score_multi.py
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

QUERIES = [
    "how to download SMOWL",
    "how do I install the SMOWL proctoring extension",
]

# Realistic shapes: KB chunks are ~500 chars (chunker.py), header + body.
PASSAGES = [
    "How to download SMOWL\n\nSMOWL is the proctoring tool used for online "
    "examinations at AMIU. Before your first proctored exam you must install the "
    "SMOWL extension in the browser you will use to sit the exam. Log in to the "
    "LMS, open the course, and select the SMOWL registration activity.",
    "How to login to LMS\n\nNavigate to the AMIU learning management system and "
    "enter the username and password issued to you at registration. If the "
    "password is rejected, use the Forgot Password link to have a reset message "
    "sent to your registered student email address.",
    "Multi-Factor Authentication setup\n\nMFA adds a second verification step to "
    "your Microsoft 365 account. Open the security settings page, choose Add "
    "sign-in method, and select Authenticator app. Scan the QR code with the "
    "Microsoft Authenticator app on your phone to finish enrolment.",
    "How to download SMOWL\n\nOnce the extension is installed, the SMOWL "
    "registration step captures a set of reference images of your face. Allow "
    "camera access when the browser prompts you, and keep your face centred and "
    "well lit while the capture runs. Registration is required only once.",
    "Student email access\n\nYour AMIU student email is hosted on Microsoft 365 "
    "and is accessible from Outlook on the web. Sign in with your full student "
    "email address as the username. Mail is retained for the duration of your "
    "enrolment on the programme.",
    "Wi-Fi connection on campus\n\nSelect the campus wireless network from the "
    "list of available networks and authenticate with your student credentials. "
    "If the connection fails, forget the network and reconnect so that the "
    "stored credentials are refreshed.",
    "How to submit an assignment\n\nOpen the course in the LMS, scroll to the "
    "assignment activity, and use Add submission. Attach the file, then confirm "
    "with Save changes. A submission receipt appears once the upload completes "
    "successfully.",
    "Exam proctoring requirements\n\nProctored examinations require a working "
    "webcam, a stable internet connection, and the SMOWL extension already "
    "registered. Sit the exam in a quiet, well-lit room and keep your student ID "
    "available for identity verification.",
]

reranker = get_reranker()
n_forms = max(settings.rerank_query_forms, 1)
queries = QUERIES[:n_forms]
passages = PASSAGES[: settings.rerank_shortlist]

print(f"model            {settings.rerank_model}")
print(f"quantize         {settings.rerank_quantize}")
print(f"torch threads    {torch.get_num_threads()}")
print(f"shortlist        {len(passages)}")
print(f"query forms      {len(queries)}")
print(f"pairs per query  {len(queries) * len(passages)}")
print()

# Warm up so neither timing pays graph-compilation cost.
reranker.score("warmup", ["warmup passage"])
reranker.score_multi(queries, passages)


def old_way() -> list[float] | None:
    """The sequential loop that used to live in retriever.py::_score_all."""
    best: list[float] | None = None
    for form in queries:
        scores = reranker.score(form, passages)
        if scores is None:
            return None
        if best is None:
            best = [float(s) for s in scores]
        else:
            best = [max(b, float(s)) for b, s in zip(best, scores)]
    return best


REPEATS = 3
old_times: list[float] = []
new_times: list[float] = []
old_scores = new_scores = None

for _ in range(REPEATS):
    t = time.perf_counter()
    old_scores = old_way()
    old_times.append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    new_scores = reranker.score_multi(queries, passages)
    new_times.append((time.perf_counter() - t) * 1000)

assert old_scores is not None, "old path returned None — reranker unavailable?"
assert new_scores is not None, "score_multi returned None — reranker unavailable?"

print(f"{'passage':<10}{'sequential':>14}{'score_multi':>14}{'delta':>12}")
worst = 0.0
for i, (a, b) in enumerate(zip(old_scores, new_scores)):
    d = abs(a - b)
    worst = max(worst, d)
    print(f"[{i}]{'':<7}{a:>14.6f}{b:>14.6f}{d:>12.2e}")

best_old = min(old_times)
best_new = min(new_times)
print()
print(f"sequential   best of {REPEATS}: {best_old:8.1f}ms  ({best_old / len(passages) / len(queries):6.1f}ms/pair)")
print(f"score_multi  best of {REPEATS}: {best_new:8.1f}ms  ({best_new / len(passages) / len(queries):6.1f}ms/pair)")
print(f"speedup                        {best_old / best_new:8.2f}x")
print()
print(f"max abs score difference: {worst:.3e}")

# The gate at rerank_min_score is an ABSOLUTE comparison on these logits, so a
# drift of even ~0.1 could flip a decline decision. Float32 reassociation across
# different batch shapes is the only expected source of difference.
TOL = 1e-3
if worst <= TOL:
    print(f"PASS — identical within {TOL:g}; the gate cannot change its mind.")
else:
    print(f"FAIL — drift {worst:.3e} exceeds {TOL:g}. Re-check the reshape in score_multi.")
    sys.exit(1)
