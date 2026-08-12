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
from backend.app.rag.domain_knowledge import get_domain_knowledge

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain knowledge — loaded from config/domain_knowledge.yaml
# ---------------------------------------------------------------------------
# The tables that used to be hardcoded here (SYNONYM_GROUPS, ABBREVIATIONS,
# ACRONYMS, SPELL_CORRECTIONS, INTENT_PATTERNS) now live in
# backend/app/rag/domain_knowledge.py as built-in defaults, overlaid by
# config/domain_knowledge.yaml at load time. Behaviour with no YAML present is
# byte-identical to before, which is what makes this a safe swap.
#
# They stay importable from this module under their old names via PEP 562
# module __getattr__. That is deliberate rather than a plain assignment: an
# import-time read would snapshot the tables before the config path is known,
# and would defeat reload_domain_knowledge() at runtime. Resolving on attribute
# access means `from ... import SYNONYM_GROUPS` always sees current data.
#
# lexical.py imports SYNONYM_GROUPS inside a function body (to break a circular
# import) and gets `list[list[str]]`, exactly as before.

_LAZY_TABLES = {
    "SYNONYM_GROUPS": lambda k: k.legacy_groups,
    "ABBREVIATIONS": lambda k: dict(k.abbreviations),
    "ACRONYMS": lambda k: dict(k.acronyms),
    "SPELL_CORRECTIONS": lambda k: dict(k.spell_corrections),
    "INTENT_PATTERNS": lambda k: [(r.patterns[0], list(r.phrasings)) for r in k.intents],
}


def __getattr__(name: str):
    """Resolve the legacy table names against the loaded domain knowledge."""
    accessor = _LAZY_TABLES.get(name)
    if accessor is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return accessor(get_domain_knowledge())


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_TABLES])


# ---------------------------------------------------------------------------
# Derived lookup tables — built once from loaded domain knowledge
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _synonym_lookup() -> dict[str, tuple[str, ...]]:
    """Lowercased term → every other term in its group(s).

    A term may appear in more than one group ("access" is both a login verb and
    a generic action); its expansion is the union of all groups containing it.
    """
    return get_domain_knowledge().synonym_lookup


@lru_cache(maxsize=1)
def _multiword_terms() -> tuple[tuple[str, ...], ...]:
    """Multi-word synonym terms, longest first, as token tuples.

    Phrase terms ("learning management system") must be matched before
    single-token ones or "system" alone would swallow the phrase and expand to
    the wrong group.
    """
    return get_domain_knowledge().phrases


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
    entities:
        Canonical names of the systems the query is about ("Moodle LMS",
        "Microsoft Authenticator"). This is what makes the Objective-1 collapse
        observable: every phrasing of "moddle" / "lms" / "course portal"
        produces the same single entity, so a benchmark can assert that
        synonymous queries were *understood* identically, not merely that they
        happened to retrieve the same chunks.
    intent_names:
        Names of the matched intent rules ("password_reset"). Drives ranking
        boosts and adaptive top_k. ``intents`` remains the human-readable
        canonical phrasing, kept for the existing debug output.
    boost_terms:
        Union of the boost vocabularies of every matched intent, lowercased.
        Consumed by the retriever's ordering stage — a boost, never a filter.
    procedural:
        True when a matched intent expects a multi-step answer. Drives the
        larger adaptive top_k so a numbered procedure is not truncated.
    """

    original: str
    normalized: str
    lexical: str
    variants: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    expansions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    intents: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    intent_names: list[str] = field(default_factory=list)
    boost_terms: tuple[str, ...] = ()
    procedural: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.corrections or self.expansions or self.intents)

    @property
    def understood(self) -> bool:
        """Whether the layer recognised anything domain-specific.

        Used by the dynamic threshold: a query we understood (known entity or
        intent) has earned a slightly more permissive gate, because the reason
        its cosine score is mediocre is usually vocabulary mismatch we have
        already corrected for — not that it is off-topic.
        """
        return bool(self.entities or self.intent_names)


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

    # Read the tables through the singleton rather than the module-level names:
    # PEP 562 __getattr__ fires on `module.NAME` from outside, not on a bare
    # global read inside the module, so the legacy names are unavailable here.
    knowledge = get_domain_knowledge()
    spell_corrections = knowledge.spell_corrections
    acronyms = knowledge.acronyms
    abbreviations = knowledge.abbreviations

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
        if key in spell_corrections:
            fixed = spell_corrections[key]
            if fixed.lower() != key:
                corrections[raw_token] = fixed
            out.append(fixed)
            continue

        # 2. Acronym canonicalisation — casing only, never a content change.
        if key in acronyms:
            out.append(acronyms[key])
            continue

        # 3. Abbreviation expansion ("pwd" → "password").
        if key in abbreviations:
            expanded = abbreviations[key]
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
) -> tuple[list[str], list[str], list[str], tuple[str, ...], bool]:
    """Build deterministic semantic paraphrases of ``normalized``.

    Returns ``(variants, intent_display_names, intent_names, boost_terms, procedural)``.
    ``variants[0]`` is always ``normalized`` so the caller can search uniformly.

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
        return [], [], [], (), False

    knowledge = get_domain_knowledge()
    variants: list[str] = [normalized]
    seen = {normalized.lower()}
    intent_display: list[str] = []
    intent_names: list[str] = []
    boost_set: set[str] = set()
    procedural = False

    def add(candidate: str) -> None:
        cand = candidate.strip()
        if not cand:
            return
        if cand.lower() in seen:
            return
        seen.add(cand.lower())
        variants.append(cand)

    # 1. Intent-based canonical phrasings.
    matched_intents = knowledge.classify_intents(normalized)
    for rule in matched_intents:
        intent_display.append(rule.phrasings[0] if rule.phrasings else rule.name)
        intent_names.append(rule.name)
        boost_set.update(t.lower() for t in rule.boost)
        if rule.procedural:
            procedural = True
        for phrasing in rule.phrasings:
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

    return (variants[: max_variants + 1], intent_display, intent_names,
            tuple(sorted(boost_set)), procedural)


def extract_entities(normalized: str) -> list[str]:
    """Canonical names of the systems ``normalized`` is about.

    This is the observable form of Objective 1's collapse: "moddle", "lms",
    "learning portal" and "course portal" all return ``["Moodle LMS"]``, so a
    test can assert that synonymous queries were *understood* the same way
    rather than inferring it from whichever chunks happened to come back.

    Only ``system`` groups count — see :attr:`SynonymGroup.is_entity`. Order is
    by first appearance in the query, and duplicates collapse, so a query
    mentioning "LMS" and "Moodle" yields one entity, not two.
    """
    knowledge = get_domain_knowledge()
    entities: list[str] = []
    for match in knowledge.match_terms(normalized):
        if not match.group.is_entity:
            continue
        if match.group.canonical not in entities:
            entities.append(match.group.canonical)
    return entities


def _canonical_for(term: str) -> Optional[str]:
    """First-listed term (rewrite form) of the first group containing ``term``."""
    knowledge = get_domain_knowledge()
    group = knowledge.group_for(term)
    return group.rewrite_form if group else None


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
        variants, intents, intent_names, boost_terms, procedural = generate_variants(
            normalized, expansions, max_variants
        )
    else:
        variants, intents, intent_names, boost_terms, procedural = (
            [normalized] if normalized else [], [], [], (), False
        )

    entities = extract_entities(normalized) if normalized else []

    processed = ProcessedQuery(
        original=original,
        normalized=normalized,
        lexical=lexical,
        variants=variants,
        corrections=corrections,
        expansions=expansions,
        intents=intents,
        entities=entities,
        intent_names=intent_names,
        boost_terms=boost_terms,
        procedural=procedural,
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
