"""Labeled retrieval evaluation set for the Amref Help Desk KB.

Ground truth is hand-labeled against the 20 articles actually present in the
knowledge base (see ``data/raw/*.json``). ``expected`` lists the article ids that
genuinely answer the query; a chunk from any of them counts as a hit.

Five query classes, because they need different verdicts and mixing them makes
both numbers meaningless:

* ``covered``  — the KB documents this. Retrieval must find it; declining is a
  failure.
* ``synonym``  — a ``covered`` question asked in vocabulary the article never
  uses (LMS↔Moodle, MFA↔2FA, SMOWL↔proctoring, portal↔SIS). Scored separately
  because this is precisely what synonym expansion exists to fix, and a drop
  here is invisible when averaged into ``covered``.
* ``typo``     — a ``covered`` question with realistic misspellings. Same
  reasoning: it isolates the fuzzy-match and spell-correction path.
* ``partial``  — the KB documents the *adjacent* topic but not the exact ask
  ("assignments" is nowhere in this KB, but LMS login is). The right behaviour is
  to answer the documented part and say plainly what is not covered. Retrieving
  the adjacent article is a hit, not a false positive.
* ``offtopic`` — nothing in the KB relates. Retrieval SHOULD return nothing and
  the model MUST decline. Confidence must be low.

``synonym`` and ``typo`` exist to make the project's targets measurable at all:
"Synonym recall ≥ 98%" and "Typo recall ≥ 95%" cannot be read off a single
blended recall number. Each entry restates a question that also appears in
``covered`` in its plain form, so a gap between the two classes is attributable
to the wording alone rather than to the topic being hard.

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

    # ---------------- synonym: right topic, wrong vocabulary ----------------
    # Each of these restates a `covered` question using a term the target article
    # never contains, so a miss isolates synonym expansion rather than topic
    # difficulty. Article 1 is titled "How to login to LMS" and says nothing
    # about "Moodle"; article 10 spells out "multi-factor authentication" and
    # never writes "2FA"; SMOWL articles rarely use the word "proctoring" alone.
    ("How do I log into Moodle?", ["1"], "synonym"),
    ("Learning Management System sign in", ["1"], "synonym"),
    ("2FA setup", ["10", "11"], "synonym"),
    ("two-factor authentication for Office 365", ["10", "11"], "synonym"),
    ("verification app enrollment", ["11", "10"], "synonym"),
    ("SIS password reset", ["9", "12", "3"], "synonym"),
    ("online exam monitoring software", ["14", "15", "18", "21"], "synonym"),
    ("webcam exam supervision", ["15", "14", "21"], "synonym"),
    ("Virtual Assessment System exam", ["14", "15", "22", "17"], "synonym"),
    ("corporate email access", ["4"], "synonym"),
    ("Microsoft 365 email login", ["4"], "synonym"),
    ("e-learning platform login", ["1"], "synonym"),

    # ---------------- typo: right topic, misspelled ----------------
    # Realistic help-desk misspellings, including the ones the TODO calls out by
    # name (moddle, athenticator, smwol). These exercise spell correction and the
    # fuzzy BM25 path; the plain spelling of each appears under `covered`.
    ("moddle login", ["1"], "typo"),
    ("athenticator app setup", ["11", "10"], "typo"),
    ("smwol camera not working", ["15", "21", "14", "16"], "typo"),
    ("reset my studnet portal pasword", ["9", "12", "3"], "typo"),
    ("microsft teams", ["5"], "typo"),
    ("suplementary exam registration", ["13"], "typo"),
    ("helpdesk contac", ["2"], "typo"),
    ("stdent email acess", ["4"], "typo"),

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

# Article ids present in the KB. 7 and 8 are absent — the source export skips
# them — so a typo'd expectation would otherwise be unsatisfiable and read as a
# retrieval failure rather than a broken label.
KB_ARTICLE_IDS = frozenset(
    {"1", "2", "3", "4", "5", "6", "9", "10", "11", "12",
     "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"}
)


def _validate() -> None:
    """Fail loudly on a malformed eval set rather than scoring against it.

    Both checks caught real mistakes while this file was being extended: a query
    duplicated across two classes (which double-counts and makes the per-class
    means disagree with the row list), and expectations naming articles that do
    not exist.
    """
    seen: dict[str, str] = {}
    for query, expected, kind in EVAL_QUERIES:
        if query in seen:
            raise ValueError(
                f"duplicate eval query {query!r} in both {seen[query]!r} and {kind!r}"
            )
        seen[query] = kind
        unknown = set(expected) - KB_ARTICLE_IDS
        if unknown:
            raise ValueError(f"{query!r} expects unknown article id(s) {sorted(unknown)}")
        if kind == "offtopic" and expected:
            raise ValueError(f"offtopic query {query!r} must expect no articles")
        if kind != "offtopic" and not expected:
            raise ValueError(f"{kind} query {query!r} must expect at least one article")


_validate()
