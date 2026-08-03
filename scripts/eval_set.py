"""Labeled retrieval evaluation set for the Amref Help Desk KB.

Ground truth is hand-labeled against the 20 articles actually present in the
knowledge base (see ``data/raw/*.json``). ``expected`` lists the article ids that
genuinely answer the query; a chunk from any of them counts as a hit.

Three query classes, because they need different verdicts and mixing them makes
both numbers meaningless:

* ``covered``  — the KB documents this. Retrieval must find it; declining is a
  failure.
* ``partial``  — the KB documents the *adjacent* topic but not the exact ask
  ("assignments" is nowhere in this KB, but LMS login is). The right behaviour is
  to answer the documented part and say plainly what is not covered. Retrieving
  the adjacent article is a hit, not a false positive.
* ``offtopic`` — nothing in the KB relates. Retrieval SHOULD return nothing and
  the model MUST decline. Confidence must be low.

``expected`` is deliberately generous (several ids per query) because more than
one article often carries the answer — e.g. SMOWL is documented across 14/15/18/21.
Recall@k asks "did we surface any article that answers this", which is the thing
that determines whether the LLM can produce a correct answer.
"""

# (query, expected_article_ids, kind)
EVAL_QUERIES: list[tuple[str, list[str], str]] = [
    # ---------------- covered: the KB documents these ----------------
    ("How do I reset my student portal password?", ["9", "12", "3"], "covered"),
    ("I forgot my portal password", ["9", "12", "3"], "covered"),
    ("What is SMOWL proctoring?", ["14", "15", "18", "21"], "covered"),
    ("smwol camera not working", ["15", "21", "14", "16"], "covered"),
    ("How do I set up Microsoft Authenticator?", ["11", "10"], "covered"),
    ("multi-factor authentication setup for Microsoft 365", ["10", "11"], "covered"),
    ("How do I log in to the LMS?", ["1"], "covered"),
    ("How do I register for supplementary exams?", ["13"], "covered"),
    ("How do I access my student email?", ["4"], "covered"),
    ("How do I use Microsoft Teams?", ["5"], "covered"),
    ("What is My Loft?", ["6"], "covered"),
    ("How do I contact the AmIU help desk?", ["2"], "covered"),
    ("How do lecturers mark exams?", ["20"], "covered"),
    ("student portal user guide", ["12", "3"], "covered"),
    ("What happens during a proctored exam?", ["21", "15", "14"], "covered"),
    ("VAS training presentation slides", ["22"], "covered"),
    ("How do I take an exam with screen monitoring?", ["15", "14", "21"], "covered"),
    ("register and take exams with proctoring software", ["14", "15"], "covered"),
    # Typo-heavy variants: these exercise the fuzzy rewrite + BM25 exact-token path.
    ("moddle login", ["1"], "covered"),
    ("athenticator app setup", ["11", "10"], "covered"),

    # ---------------- partial: adjacent coverage only ----------------
    # "assignments" appears in no article in this KB. LMS login is the nearest
    # documented thing, so surfacing article 1 is the correct outcome and the
    # answer should say assignments specifically are not documented.
    ("I cannot access my assignments", ["1"], "partial"),
    ("moddle assignments", ["1"], "partial"),
    ("Where do I check my grades?", ["1", "3", "12"], "partial"),

    # ---------------- offtopic: must return nothing and decline ----------------
    ("What is the capital of France?", [], "offtopic"),
    ("How do I bake a chocolate cake?", [], "offtopic"),
    ("Who won the 2022 World Cup?", [], "offtopic"),
]
