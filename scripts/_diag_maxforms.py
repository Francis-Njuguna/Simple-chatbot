"""Does max-over-query-forms fix synonym recall WITHOUT leaking off-topic?

The gate currently depends on which surface form the user typed: "LMS login"
scores +7.6 against the login article, "Moodle login" scores -8.7 against the
SAME passages. Taking the max over several fluent phrasings of the question
should rescue the second case.

The risk is that scoring N phrasings gives an off-topic query N chances to
sneak past the gate. This measures both sides.
"""

import sys

sys.path.insert(0, ".")

import anyio  # noqa: E402

ON_TOPIC = [
    "How do I log into Moodle?", "LMS login", "Can't access LMS",
    "moddle login", "Learning Management System login",
    "Student email", "Outlook login", "University email",
    "Authenticator setup", "MFA", "2FA", "Microsoft Authenticator",
    "VAS exam", "Assessment system", "Online exam",
    "SMOWL camera", "Proctoring software", "Exam monitoring",
    "portal pwd", "I forgot my password", "can't login",
]
OFF_TOPIC = [
    "What is the capital of France?", "How do I bake sourdough bread?",
    "Who won the world cup in 2018?", "What is the weather tomorrow?",
    "Explain quantum entanglement", "Best pizza recipe",
    "How do I change my car tyre?", "Python list comprehension syntax",
]


async def main():
    from backend.app.rag.retriever import get_retriever
    from backend.app.rag.reranker import get_reranker
    from backend.app.rag.query_processing import process_query
    from backend.app.rag.lexical import get_lexical_index
    from backend.app.config import get_settings
    from backend.app.database.chroma import query_text_collection

    settings = get_settings()
    gate = settings.rerank_min_score
    r = get_retriever()
    rr = get_reranker()
    vocab = get_lexical_index().vocabulary()

    async def best_scores(q, n_forms):
        """Top rerank score for q using 1 form vs n_forms (max-pooled)."""
        p = process_query(q, fuzzy_vocabulary=vocab)
        emb = await r.embedding_service.embed_query_async(p.normalized)
        res = query_text_collection(query_embedding=emb, n_results=8)
        docs = res["documents"][0]
        if not docs:
            return None, None
        forms = [p.normalized] + [v for v in p.variants if v != p.normalized]
        forms = forms[:n_forms]
        best = [-99.0] * len(docs)
        for f in forms:
            for i, s in enumerate(rr.score(f, docs)):
                best[i] = max(best[i], float(s))
        return max(best), forms

    for label, queries, want in (
        ("ON-TOPIC (want: passes gate)", ON_TOPIC, True),
        ("OFF-TOPIC (want: blocked)", OFF_TOPIC, False),
    ):
        print(f"\n{'=' * 74}\n{label}   gate={gate}\n{'=' * 74}")
        for n in (1, 2, 3):
            ok = 0
            details = []
            for q in queries:
                top, _ = await best_scores(q, n)
                passed = top is not None and top >= gate
                if passed == want:
                    ok += 1
                else:
                    details.append(f"{q!r}({top:.1f})")
            pct = 100 * ok / len(queries)
            print(f"  {n} form(s): {ok:2d}/{len(queries)} correct ({pct:5.1f}%)")
            if details:
                print(f"           wrong: {', '.join(details[:6])}")


anyio.run(main)
