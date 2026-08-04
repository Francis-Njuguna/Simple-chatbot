"""Query preprocessing: normalization, synonym expansion, multi-query generation.

This module sits at the front of the retrieval pipeline and turns one user query
into several retrieval-optimised forms. The central design decision is that
**different stages of retrieval want different text**, so we produce all of them
up front rather than mutating one string in place:

* ``normalized`` — lowercased, de-punctuated, spell-corrected, abbreviations
  expanded. Feeds the **vector** search.
* ``lexical``    — normalized *plus* every domain synonym appended. Feeds
  **BM25** only.
* ``variants``   — 3-5 semantic paraphrases. Each is embedded and searched, and
  the results are fused, so retrieval no longer depends on the user's phrasing.
* ``original``   — the user's verbatim question. Feeds the **cross-encoder**.

Why the cross-encoder must keep the original
--------------------------------------------
Synonym expansion is the right move for BM25 (more exact tokens to match) and
disastrous for a cross-encoder. A cross-encoder is trained on natural
question/passage pairs; handing it ``"lms moodle learning management system
login sign in access log on"`` scores like keyword soup and its logits stop
meaning "does this passage answer the question". Since that logit is the gate
protecting off-topic precision, corrupting it would trade away exactly the
metric we must not lose.

So expansion widens the *candidate pool*, and judgement stays with the original
question. That split is what lets recall rise without precision falling — the
two goals that otherwise pull against each other.

Why variants are generated deterministically by default
-------------------------------------------------------
An LLM paraphrase call costs 1-2s and makes retrieval non-reproducible, which
breaks benchmarking (the same query would score differently run to run) and eats
most of the 3s latency budget. Rule-based variants — synonym substitution plus
intent templates — cost microseconds, are deterministic, and cover the phrasing
space this KB actually sees. ``MULTI_QUERY_USE_LLM=true`` switches to LLM
paraphrases when a query genuinely needs open-ended rewriting.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain synonym groups
# ---------------------------------------------------------------------------
# Each list holds terms that refer to the same thing *in this knowledge base*.
# "LMS" and "Moodle" are not dictionary synonyms; here they name one system, and
# a user typing either must reach article 1. Groups are intentionally generous —
# the cross-encoder gate downstream removes anything that does not actually
# answer the question, so over-inclusion costs recall nothing and precision
# nothing.
#
# Order matters only for display: the first entry is treated as the canonical
# form when logging an expansion.
SYNONYM_GROUPS: list[list[str]] = [
    # --- Systems and platforms ---
    ["LMS", "Moodle", "Learning Management System", "learning platform", "e-learning"],
    ["Student Portal", "portal", "SIS", "student information system", "student system"],
    [
        "MFA", "Microsoft Authenticator", "authenticator", "2FA",
        "two-factor authentication", "multi-factor authentication",
        "multifactor authentication", "authenticator app", "verification app",
    ],
    [
        "student email", "university email", "corporate email", "Outlook",
        "Microsoft 365 email", "M365 email", "Office 365 email",
        "official email", "mcampus email", "campus email",
    ],
    [
        "VAS", "Virtual Assessment System", "assessment platform",
        "assessment system", "exam system", "online exam", "online examination",
    ],
    [
        "SMOWL", "proctoring", "online exam monitoring", "screen monitoring",
        "proctoring software", "exam monitoring", "proctored exam",
        "remote proctoring", "exam supervision",
    ],
    ["Microsoft Teams", "Teams", "MS Teams"],
    ["My Loft", "MyLoft", "library resources", "digital library"],
    ["help desk", "helpdesk", "support", "IT support", "technical support"],

    # --- Actions and intents ---
    ["login", "log in", "sign in", "signin", "log on", "logon", "access"],
    ["password", "passcode", "login credentials", "credentials"],
    ["reset", "recover", "change", "retrieve", "forgot"],
    ["register", "registration", "enroll", "enrolment", "enrollment", "sign up"],
    ["exam", "examination", "test", "assessment"],
    ["supplementary exam", "special exam", "resit", "retake"],
    ["camera", "webcam", "cam", "video"],
    ["grades", "marks", "results", "scores", "transcript"],
    ["setup", "set up", "configure", "install", "installation"],
]


# Token → canonical expansion, applied before synonym lookup. These are
# abbreviations and clippings, not misspellings: the user meant to type them.
ABBREVIATIONS: dict[str, str] = {
    "pwd": "password",
    "pass": "password",
    "pw": "password",
    "auth": "authentication",
    "authn": "authentication",
    "cam": "camera",
    "acct": "account",
    "acc": "account",
    "reg": "registration",
    "info": "information",
    "admin": "administrator",
    "uni": "university",
    "msg": "message",
    "config": "configuration",
    "docs": "documentation",
    "app": "application",
    "num": "number",
    "id": "identification",
    "faq": "frequently asked questions",
    "kb": "knowledge base",
    "sms": "text message",
}


# Acronyms that must survive normalization intact and in canonical casing. BM25
# matches exact tokens, so "smowl" and "SMOWL" must not be two different things.
ACRONYMS: dict[str, str] = {
    "lms": "LMS",
    "vas": "VAS",
    "mfa": "MFA",
    "2fa": "2FA",
    "sis": "SIS",
    "smowl": "SMOWL",
    "m365": "M365",
    "amiu": "AmIU",
    "it": "IT",
    "ms": "MS",
    "pdf": "PDF",
    "otp": "OTP",
}


# Observed (and next-most-likely) misspellings of domain terms, mapped to the
# canonical form. Fuzzy matching cannot reach these reliably: "smwol" is a
# transposition at SequenceMatcher ratio 0.80 and "moddle" one edit at 0.83 —
# both below any bar loose enough to be safe for ordinary words. An explicit map
# is both safer and faster than lowering the fuzzy threshold.
SPELL_CORRECTIONS: dict[str, str] = {
    # SMOWL
    "smwol": "SMOWL", "swoml": "SMOWL", "swowl": "SMOWL", "smowll": "SMOWL",
    "smowel": "SMOWL", "smol": "SMOWL", "smowl": "SMOWL", "smowal": "SMOWL",
    # Moodle
    "moddle": "Moodle", "modle": "Moodle", "moodel": "Moodle",
    "mooddle": "Moodle", "moodle": "Moodle", "muddle": "Moodle",
    # Authenticator
    "athenticator": "authenticator", "authentificator": "authenticator",
    "autheticator": "authenticator", "authenicator": "authenticator",
    "authenticater": "authenticator", "athenticater": "authenticator",
    # Password
    "pasword": "password", "paswword": "password", "passwrod": "password",
    "passowrd": "password", "pssword": "password", "passwd": "password",
    # Portal
    "protal": "portal", "porta": "portal", "portall": "portal",
    # Microsoft
    "micorsoft": "Microsoft", "microsft": "Microsoft", "mircosoft": "Microsoft",
    # Outlook
    "outllok": "Outlook", "otlook": "Outlook", "outlok": "Outlook",
    # Login
    "logn": "login", "loign": "login", "lgoin": "login",
    # Exam
    "exame": "exam", "exm": "exam", "examm": "exam",
    # Register
    "registor": "register", "regsiter": "register", "reigster": "register",
    # Teams
    "taems": "Teams", "tems": "Teams",
    # Supplementary
    "suplementary": "supplementary", "supplimentary": "supplementary",
}


# Intent patterns → canonical phrasings. This is what makes "can't login",
# "unable to access", "where do I sign in" and "login problem" retrieve the same
# article: each matches the login-trouble intent and contributes the same
# canonical variants, regardless of how the user phrased it.
#
# (compiled pattern, [canonical phrasings to add as variants])
INTENT_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (
        re.compile(r"\b(forgot|lost|forgotten|don'?t remember|cannot remember)\b.*\b(password|pwd|passcode)\b|"
                   r"\b(password|pwd)\b.*\b(forgot|lost|reset|recover|change)\b", re.I),
        ["How do I reset my password?", "password reset steps", "recover forgotten password"],
    ),
    (
        re.compile(r"\b(can'?t|cannot|unable to|couldn'?t|failed to|having (trouble|issues?|problems?))\b.*"
                   r"\b(log ?in|log ?on|sign ?in|access|get in|enter)\b|"
                   r"\b(login|log ?in|sign ?in)\b.*\b(problem|issue|error|trouble|fail(ed|ure)?|not working)\b",
                   re.I),
        ["How do I log in?", "login troubleshooting", "cannot sign in to my account"],
    ),
    (
        re.compile(r"\bwhere\b.*\b(do|can|should)\b.*\b(i|we)\b.*\b(log ?in|sign ?in|access|find)\b", re.I),
        ["How do I log in?", "where to sign in", "login page location"],
    ),
    (
        re.compile(r"\b(how|steps?|guide|instructions?|procedure|process)\b.*\b(set ?up|setup|configure|install)\b", re.I),
        ["setup instructions", "configuration steps", "how to set up"],
    ),
    (
        re.compile(r"\b(camera|webcam|cam|microphone|mic)\b.*\b(not working|fail(ed|ing)?|issue|problem|error|black|blank)\b|"
                   r"\b(not working|problem|issue)\b.*\b(camera|webcam|cam)\b", re.I),
        ["camera not working during exam", "webcam troubleshooting", "fix camera detection"],
    ),
    (
        re.compile(r"\b(what is|what'?s|explain|describe|tell me about|meaning of)\b", re.I),
        ["overview and explanation", "what it is and how it works"],
    ),
    (
        re.compile(r"\b(register|registration|enroll|sign ?up)\b.*\b(exam|examination|test|assessment|course|unit)\b|"
                   r"\b(exam|examination)\b.*\b(register|registration|enroll)\b", re.I),
        ["How do I register for exams?", "exam registration process"],
    ),
    (
        re.compile(r"\b(contact|reach|call|speak to|talk to|get help from)\b.*"
                   r"\b(help ?desk|support|IT|admin|administrator|staff)\b", re.I),
        ["How do I contact the help desk?", "support contact details"],
    ),
]


# ---------------------------------------------------------------------------
# Derived lookup tables — built once
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _synonym_lookup() -> dict[str, tuple[str, ...]]:
    """Lowercased term → every other term in its group(s).

    A term may appear in more than one group ("access" is both a login verb and
    a generic action); its expansion is the union of all groups containing it.
    """
    lookup: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        for term in group:
            key = term.lower()
            lookup.setdefault(key, set()).update(
                other for other in group if other.lower() != key
            )
    return {key: tuple(sorted(vals)) for key, vals in lookup.items()}


@lru_cache(maxsize=1)
def _multiword_terms() -> tuple[tuple[str, ...], ...]:
    """Multi-word synonym terms, longest first, as token tuples.

    Phrase terms ("learning management system") must be matched before
    single-token ones or "system" alone would swallow the phrase and expand to
    the wrong group.
    """
    phrases = [
        tuple(t.lower().split())
        for group in SYNONYM_GROUPS
        for t in group
        if " " in t or "-" in t
    ]
    # Hyphens split too, so "two-factor authentication" matches "two factor
    # authentication" as typed.
    normalised = [tuple(" ".join(p).replace("-", " ").split()) for p in phrases]
    return tuple(sorted(set(normalised), key=len, reverse=True))


# Punctuation to strip. Apostrophes and intra-word hyphens survive so "student's"
# and "re-register" stay single tokens, matching the BM25 tokeniser in lexical.py.
_PUNCT_RE = re.compile(r"[^\w\s'\-]+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", re.I)


@dataclass
class ProcessedQuery:
    """Every form of the query the retrieval stages need.

    Attributes
    ----------
    original:
        Verbatim user input. Used for cross-encoder scoring — see module
        docstring for why this must not be the expanded form.
    normalized:
        Lowercased, de-punctuated, spell-corrected, abbreviations expanded.
        Used for the vector query.
    lexical:
        ``normalized`` plus all synonym terms. BM25 only.
    variants:
        Semantic paraphrases including ``normalized`` at index 0. Each is
        embedded and searched separately; results are fused.
    corrections / expansions / intents:
        Diagnostics for the debug log — what the pipeline actually changed.
    """

    original: str
    normalized: str
    lexical: str
    variants: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    expansions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    intents: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.corrections or self.expansions or self.intents)


# ---------------------------------------------------------------------------
# Stage 1: normalization
# ---------------------------------------------------------------------------

def normalize_query(
    query: str,
    fuzzy_vocabulary: Optional[Iterable[str]] = None,
    fuzzy_threshold: float = 0.86,
) -> tuple[str, dict[str, str]]:
    """Lowercase, strip punctuation, fix spelling, expand abbreviations.

    Returns ``(normalized_query, {original_token: replacement})``.

    ``fuzzy_vocabulary`` is the KB's own token set. When supplied, a token that
    survives the explicit maps but closely matches a corpus token is corrected
    to it — this catches misspellings we never enumerated. The threshold stays
    high (0.86 ≈ one edit in seven characters) because a wrong "correction"
    silently retrieves the wrong article, which is worse than no correction.
    """
    if not query or not query.strip():
        return "", {}

    corrections: dict[str, str] = {}

    # Strip punctuation first so "moodle?" and "moodle" normalise identically.
    cleaned = _PUNCT_RE.sub(" ", query)
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    out: list[str] = []
    for raw_token in cleaned.split():
        # Compare on a bare alphanumeric key; keep the raw token for reporting.
        key = raw_token.lower().strip("'-")
        if not key:
            continue

        # 1. Explicit spelling correction (highest confidence).
        if key in SPELL_CORRECTIONS:
            fixed = SPELL_CORRECTIONS[key]
            if fixed.lower() != key:
                corrections[raw_token] = fixed
            out.append(fixed)
            continue

        # 2. Acronym canonicalisation — casing only, never a content change.
        if key in ACRONYMS:
            out.append(ACRONYMS[key])
            continue

        # 3. Abbreviation expansion ("pwd" → "password").
        if key in ABBREVIATIONS:
            expanded = ABBREVIATIONS[key]
            corrections[raw_token] = expanded
            out.append(expanded)
            continue

        # 4. Fuzzy correction against the KB's own vocabulary, last resort.
        if fuzzy_vocabulary is not None and len(key) >= 4 and key not in fuzzy_vocabulary:
            match = _fuzzy_match(key, fuzzy_vocabulary, fuzzy_threshold)
            if match is not None:
                corrections[raw_token] = match
                out.append(match)
                continue

        out.append(key)

    normalized = " ".join(out)
    return normalized, corrections


def _fuzzy_match(
    token: str, vocabulary: Iterable[str], threshold: float
) -> Optional[str]:
    """Closest vocabulary term above ``threshold``, or None.

    Uses difflib rather than a Levenshtein package to avoid a dependency for a
    path that runs on a handful of tokens per query. ``get_close_matches`` is
    C-backed and pre-filters by length, so this stays sub-millisecond.
    """
    import difflib

    matches = difflib.get_close_matches(token, vocabulary, n=1, cutoff=threshold)
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Stage 2: synonym expansion
# ---------------------------------------------------------------------------

def expand_synonyms(normalized: str) -> tuple[str, dict[str, tuple[str, ...]]]:
    """Append every domain synonym of every matched term.

    Returns ``(expanded_text, {matched_term: synonyms_added})``.

    The result is for **BM25 only**. Multi-word phrases are matched before
    single tokens so "learning management system" expands as one unit rather
    than as three unrelated words.
    """
    if not normalized:
        return "", {}

    lookup = _synonym_lookup()
    tokens = normalized.lower().split()
    consumed = [False] * len(tokens)
    expansions: dict[str, tuple[str, ...]] = {}

    # Phrases first, longest to shortest.
    for phrase in _multiword_terms():
        n = len(phrase)
        if n > len(tokens):
            continue
        for i in range(len(tokens) - n + 1):
            if any(consumed[i : i + n]):
                continue
            if tuple(tokens[i : i + n]) == phrase:
                term = " ".join(phrase)
                if term in lookup:
                    expansions[term] = lookup[term]
                    for j in range(i, i + n):
                        consumed[j] = True

    # Then single tokens not already covered by a phrase match.
    for i, token in enumerate(tokens):
        if consumed[i]:
            continue
        if token in lookup:
            expansions[token] = lookup[token]

    if not expansions:
        return normalized, {}

    # Deduplicate additions case-insensitively against the query itself so a
    # term already present is not repeated (BM25 term frequency would otherwise
    # over-weight it).
    present = set(tokens)
    additions: list[str] = []
    for syns in expansions.values():
        for syn in syns:
            low = syn.lower()
            if low in present:
                continue
            present.add(low)
            additions.append(syn)

    return f"{normalized} {' '.join(additions)}".strip(), expansions


# ---------------------------------------------------------------------------
# Stage 3: multi-query variant generation
# ---------------------------------------------------------------------------

def generate_variants(
    normalized: str,
    expansions: dict[str, tuple[str, ...]],
    max_variants: int = 4,
) -> tuple[list[str], list[str]]:
    """Build deterministic semantic paraphrases of ``normalized``.

    Returns ``(variants, matched_intents)``. ``variants[0]`` is always
    ``normalized`` so the caller can search the list uniformly.

    Three sources, in priority order:

    1. **Intent templates** — canonical phrasings for a recognised intent. This
       is what collapses "can't login" / "unable to access" / "where do I sign
       in" onto the same retrieval, since all three match one pattern and
       inherit its phrasings.
    2. **Synonym substitution** — swap a matched term for its canonical
       alternative ("lms login" → "Moodle login"). Targets the vector search,
       which responds to a coherent rephrasing far better than to a bag of
       appended synonyms.
    3. **Question reframing** — a bare keyword query ("MFA") becomes a natural
       question, which is the register the corpus is written in and what the
       bi-encoder was trained on.

    All are string operations: no LLM, no network, reproducible run to run.
    """
    if not normalized:
        return [], []

    variants: list[str] = [normalized]
    seen = {normalized.lower()}
    intents: list[str] = []

    def add(candidate: str) -> None:
        cand = candidate.strip()
        if not cand:
            return
        if cand.lower() in seen:
            return
        seen.add(cand.lower())
        variants.append(cand)

    # 1. Intent-based canonical phrasings.
    for pattern, phrasings in INTENT_PATTERNS:
        if pattern.search(normalized):
            intents.append(phrasings[0])
            for phrasing in phrasings:
                add(phrasing)

    # 2. Synonym substitution — one coherent rewrite per matched term, using the
    # group's canonical (first-listed) form. Prefer expanding acronyms to their
    # long form, which carries far more signal for a bi-encoder.
    for term, syns in expansions.items():
        if not syns:
            continue
        canonical = _canonical_for(term)
        if canonical and canonical.lower() != term.lower():
            add(_substitute(normalized, term, canonical))
        # Longest synonym is usually the spelled-out name ("Learning Management
        # System" over "LMS") — the most informative single substitution.
        longest = max(syns, key=len)
        if longest.lower() != term.lower() and longest != canonical:
            add(_substitute(normalized, term, longest))

    # 3. Question reframing for terse keyword queries.
    word_count = len(normalized.split())
    if word_count <= 3 and not re.match(r"^\s*(how|what|where|when|why|who|can|do|is)\b",
                                        normalized, re.I):
        add(f"How do I use {normalized}?")
        add(f"What is {normalized}?")

    return variants[: max_variants + 1], intents


def _canonical_for(term: str) -> Optional[str]:
    """First-listed term of the first group containing ``term``."""
    low = term.lower()
    for group in SYNONYM_GROUPS:
        if any(t.lower() == low for t in group):
            return group[0]
    return None


def _substitute(text: str, term: str, replacement: str) -> str:
    """Replace whole-word ``term`` with ``replacement``, case-insensitively."""
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.I)
    return pattern.sub(replacement, text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_query(
    query: str,
    *,
    enable_normalization: bool = True,
    enable_synonyms: bool = True,
    enable_multi_query: bool = True,
    max_variants: int = 4,
    fuzzy_vocabulary: Optional[Iterable[str]] = None,
) -> ProcessedQuery:
    """Run the full preprocessing pipeline over ``query``.

    Every stage is independently disablable so the benchmark can attribute a
    recall change to one specific stage rather than to "preprocessing".
    """
    original = query or ""

    if not enable_normalization:
        normalized, corrections = original.strip().lower(), {}
    else:
        normalized, corrections = normalize_query(original, fuzzy_vocabulary)

    if enable_synonyms:
        lexical, expansions = expand_synonyms(normalized)
    else:
        lexical, expansions = normalized, {}

    if enable_multi_query:
        variants, intents = generate_variants(normalized, expansions, max_variants)
    else:
        variants, intents = ([normalized] if normalized else []), []

    processed = ProcessedQuery(
        original=original,
        normalized=normalized,
        lexical=lexical,
        variants=variants,
        corrections=corrections,
        expansions=expansions,
        intents=intents,
    )

    if processed.changed:
        logger.debug(
            "[query] %r → normalized=%r variants=%d corrections=%s expansions=%s",
            original, normalized, len(variants),
            corrections or "-", list(expansions) or "-",
        )
    return processed


def reset_caches() -> None:
    """Drop derived lookup tables — for tests that mutate the synonym groups."""
    _synonym_lookup.cache_clear()
    _multiword_terms.cache_clear()
