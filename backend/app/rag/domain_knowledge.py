"""Domain knowledge: synonyms, entities and intents — configurable via YAML.

Why this module exists
----------------------
Retrieval quality for this knowledge base rests on facts no embedding model
knows: that "LMS" and "Moodle" name one system *here*, that "SMOWL" is the
proctoring tool, that a user typing "smwol" meant it. Those facts change as the
KB grows, and they are maintained by whoever runs the help desk — not by
whoever ships the code. So they live in ``config/domain_knowledge.yaml`` and are
loaded at startup.

Backward compatibility is the load rule, not an afterthought: every table here
has a built-in default identical to what used to be hardcoded in
``query_processing``. The YAML file *overlays* those defaults rather than
replacing them, so a deployment without the file behaves exactly as it did
before this module existed. ``tests/test_domain_knowledge.py`` asserts the
shipped YAML is a superset of the built-ins, which is what stops the two copies
drifting apart.

One table, two jobs
-------------------
A synonym group doubles as an *entity*. The group holding "LMS", "Moodle" and
"learning portal" is the entity whose canonical name is "Moodle LMS", so
recognising a synonym and extracting an entity are one lookup instead of two
parallel tables that disagree six months from now.

That is what makes these collapse onto one retrieval::

    moddle · moodle · lms · learning portal · course portal   →  Moodle LMS
    2FA · MFA · authenticator · verification app              →  Microsoft Authenticator

Phrase matching
---------------
:meth:`DomainKnowledge.match_terms` is the single place multi-word terms are
resolved, longest-first. That ordering is load-bearing: matching single tokens
first would let "system" alone swallow "learning management system" and expand
into the wrong group.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

GroupKind = Literal["system", "action", "object"]


# ---------------------------------------------------------------------------
# Built-in defaults
# ---------------------------------------------------------------------------
# These are the tables that were previously hardcoded in query_processing.py,
# verbatim. They are the fallback when config/domain_knowledge.yaml is absent
# or unreadable, so the retrieval pipeline never depends on a file existing.
#
# `terms[0]` is the form used when rewriting a query variant, NOT `canonical`.
# It is deliberately the KB's own preferred wording ("LMS", because the article
# is titled "How to login to LMS"), which is often shorter than the entity's
# display name. Reordering these lists changes measured recall — see the
# variant-order note in retriever.py Stage 5.

_BUILTIN_SYNONYMS: dict[str, dict[str, Any]] = {
    "lms": {
        "canonical": "Moodle LMS",
        "kind": "system",
        "terms": ["LMS", "Moodle", "Learning Management System",
                  "learning platform", "e-learning"],
    },
    "student_portal": {
        "canonical": "Student Portal",
        "kind": "system",
        "terms": ["Student Portal", "portal", "SIS",
                  "student information system", "student system"],
    },
    "mfa": {
        "canonical": "Microsoft Authenticator",
        "kind": "system",
        "terms": ["MFA", "Microsoft Authenticator", "authenticator", "2FA",
                  "two-factor authentication", "multi-factor authentication",
                  "multifactor authentication", "authenticator app",
                  "verification app"],
    },
    "email": {
        "canonical": "University Email",
        "kind": "system",
        "terms": ["student email", "university email", "corporate email",
                  "Outlook", "Microsoft 365 email", "M365 email",
                  "Office 365 email", "official email", "mcampus email",
                  "campus email"],
    },
    "vas": {
        "canonical": "Virtual Assessment System",
        "kind": "system",
        "terms": ["VAS", "Virtual Assessment System", "assessment platform",
                  "assessment system", "exam system", "online exam",
                  "online examination"],
    },
    "smowl": {
        "canonical": "SMOWL Proctoring",
        "kind": "system",
        "terms": ["SMOWL", "proctoring", "online exam monitoring",
                  "screen monitoring", "proctoring software", "exam monitoring",
                  "proctored exam", "remote proctoring", "exam supervision"],
    },
    "teams": {
        "canonical": "Microsoft Teams",
        "kind": "system",
        "terms": ["Microsoft Teams", "Teams", "MS Teams"],
    },
    "myloft": {
        "canonical": "My Loft",
        "kind": "system",
        "terms": ["My Loft", "MyLoft", "library resources", "digital library"],
    },
    "helpdesk": {
        "canonical": "Help Desk",
        "kind": "system",
        "terms": ["help desk", "helpdesk", "support", "IT support",
                  "technical support"],
    },
    "login": {
        "canonical": "Login",
        "kind": "action",
        "terms": ["login", "log in", "sign in", "signin", "log on", "logon",
                  "access"],
    },
    "password": {
        "canonical": "Password",
        "kind": "object",
        "terms": ["password", "passcode", "login credentials", "credentials"],
    },
    "reset": {
        "canonical": "Reset",
        "kind": "action",
        "terms": ["reset", "recover", "change", "retrieve", "forgot"],
    },
    "register": {
        "canonical": "Registration",
        "kind": "action",
        "terms": ["register", "registration", "enroll", "enrolment",
                  "enrollment", "sign up"],
    },
    "exam": {
        "canonical": "Exam",
        "kind": "object",
        "terms": ["exam", "examination", "test", "assessment"],
    },
    "supplementary_exam": {
        "canonical": "Supplementary Exam",
        "kind": "object",
        "terms": ["supplementary exam", "special exam", "resit", "retake"],
    },
    "camera": {
        "canonical": "Camera",
        "kind": "object",
        "terms": ["camera", "webcam", "cam", "video"],
    },
    "grades": {
        "canonical": "Grades",
        "kind": "object",
        "terms": ["grades", "marks", "results", "scores", "transcript"],
    },
    "setup": {
        "canonical": "Setup",
        "kind": "action",
        "terms": ["setup", "set up", "configure", "install", "installation"],
    },
}


_BUILTIN_ABBREVIATIONS: dict[str, str] = {
    "pwd": "password", "pass": "password", "pw": "password",
    "auth": "authentication", "authn": "authentication", "cam": "camera",
    "acct": "account", "acc": "account", "reg": "registration",
    "info": "information", "admin": "administrator", "uni": "university",
    "msg": "message", "config": "configuration", "docs": "documentation",
    "app": "application", "num": "number", "id": "identification",
    "faq": "frequently asked questions", "kb": "knowledge base",
    "sms": "text message",
}


_BUILTIN_ACRONYMS: dict[str, str] = {
    "lms": "LMS", "vas": "VAS", "mfa": "MFA", "2fa": "2FA", "sis": "SIS",
    "smowl": "SMOWL", "m365": "M365", "amiu": "AmIU", "it": "IT", "ms": "MS",
    "pdf": "PDF", "otp": "OTP",
}


_BUILTIN_SPELL_CORRECTIONS: dict[str, str] = {
    "smwol": "SMOWL", "swoml": "SMOWL", "swowl": "SMOWL", "smowll": "SMOWL",
    "smowel": "SMOWL", "smol": "SMOWL", "smowl": "SMOWL", "smowal": "SMOWL",
    "moddle": "Moodle", "modle": "Moodle", "moodel": "Moodle",
    "mooddle": "Moodle", "moodle": "Moodle", "muddle": "Moodle",
    "athenticator": "authenticator", "authentificator": "authenticator",
    "autheticator": "authenticator", "authenicator": "authenticator",
    "authenticater": "authenticator", "athenticater": "authenticator",
    "pasword": "password", "paswword": "password", "passwrod": "password",
    "passowrd": "password", "pssword": "password", "passwd": "password",
    "protal": "portal", "porta": "portal", "portall": "portal",
    "micorsoft": "Microsoft", "microsft": "Microsoft", "mircosoft": "Microsoft",
    "outllok": "Outlook", "otlook": "Outlook", "outlok": "Outlook",
    "logn": "login", "loign": "login", "lgoin": "login",
    "exame": "exam", "exm": "exam", "examm": "exam",
    "registor": "register", "regsiter": "register", "reigster": "register",
    "taems": "Teams", "tems": "Teams",
    "suplementary": "supplementary", "supplimentary": "supplementary",
}


# name → (patterns, phrasings, boost, procedural)
_BUILTIN_INTENTS: dict[str, dict[str, Any]] = {
    "password_reset": {
        "patterns": [
            r"\b(forgot|lost|forgotten|don'?t remember|cannot remember)\b.*\b(password|pwd|passcode)\b",
            r"\b(password|pwd)\b.*\b(forgot|lost|reset|recover|change)\b",
        ],
        "phrasings": ["How do I reset my password?", "password reset steps",
                      "recover forgotten password"],
        "boost": ["password", "reset", "credentials", "authentication",
                  "portal", "recover", "login"],
        "procedural": True,
    },
    "login_trouble": {
        "patterns": [
            r"\b(can'?t|cannot|unable to|couldn'?t|failed to|having (trouble|issues?|problems?))\b.*"
            r"\b(log ?in|log ?on|sign ?in|access|get in|enter)\b",
            r"\b(login|log ?in|sign ?in)\b.*\b(problem|issue|error|trouble|fail(ed|ure)?|not working)\b",
        ],
        "phrasings": ["How do I log in?", "login troubleshooting",
                      "cannot sign in to my account"],
        "boost": ["login", "sign in", "access", "password", "portal",
                  "account", "credentials"],
        "procedural": True,
    },
    "login_location": {
        "patterns": [
            r"\bwhere\b.*\b(do|can|should)\b.*\b(i|we)\b.*\b(log ?in|sign ?in|access|find)\b",
        ],
        "phrasings": ["How do I log in?", "where to sign in",
                      "login page location"],
        "boost": ["login", "sign in", "url", "address", "portal", "access"],
        "procedural": False,
    },
    "setup_howto": {
        "patterns": [
            r"\b(how|steps?|guide|instructions?|procedure|process)\b.*\b(set ?up|setup|configure|install)\b",
        ],
        "phrasings": ["setup instructions", "configuration steps",
                      "how to set up"],
        "boost": ["setup", "configure", "install", "steps", "download",
                  "enrol"],
        "procedural": True,
    },
    "camera_issue": {
        "patterns": [
            r"\b(camera|webcam|cam|microphone|mic)\b.*"
            r"\b(not working|fail(ed|ing)?|issue|problem|error|black|blank)\b",
            r"\b(not working|problem|issue)\b.*\b(camera|webcam|cam)\b",
        ],
        "phrasings": ["camera not working during exam", "webcam troubleshooting",
                      "fix camera detection"],
        "boost": ["camera", "webcam", "SMOWL", "proctoring", "permissions",
                  "browser", "exam"],
        "procedural": True,
    },
    "definition": {
        "patterns": [r"\b(what is|what'?s|explain|describe|tell me about|meaning of)\b"],
        "phrasings": ["overview and explanation", "what it is and how it works"],
        "boost": [],
        "procedural": False,
    },
    "exam_registration": {
        "patterns": [
            r"\b(register|registration|enroll|sign ?up)\b.*"
            r"\b(exam|examination|test|assessment|course|unit)\b",
            r"\b(exam|examination)\b.*\b(register|registration|enroll)\b",
        ],
        "phrasings": ["How do I register for exams?", "exam registration process"],
        "boost": ["exam", "registration", "register", "VAS", "assessment",
                  "deadline"],
        "procedural": True,
    },
    "contact_support": {
        "patterns": [
            r"\b(contact|reach|call|speak to|talk to|get help from)\b.*"
            r"\b(help ?desk|support|IT|admin|administrator|staff)\b",
        ],
        "phrasings": ["How do I contact the help desk?", "support contact details"],
        "boost": ["help desk", "support", "contact", "email", "phone", "ticket"],
        "procedural": False,
    },
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynonymGroup:
    """A set of surface forms that mean one thing in this knowledge base."""

    id: str
    canonical: str
    kind: str
    terms: tuple[str, ...]

    @property
    def rewrite_form(self) -> str:
        """The form used when rewriting a query variant.

        ``terms[0]``, not ``canonical``: the KB's own preferred wording is what
        the articles are written in, and that is what the bi-encoder and the
        cross-encoder both respond to.
        """
        return self.terms[0] if self.terms else self.canonical

    @property
    def is_entity(self) -> bool:
        """Whether matching this group means the query is *about* something.

        Only ``system`` groups qualify. A query mentioning "login" is not about
        an entity — nearly every article in this KB mentions logging in — so
        treating action and object groups as entities would make entity-based
        boosting fire on everything and mean nothing.
        """
        return self.kind == "system"


@dataclass(frozen=True)
class IntentRule:
    """What the user is trying to do, independent of how they worded it."""

    name: str
    patterns: tuple[re.Pattern[str], ...]
    phrasings: tuple[str, ...]
    boost: tuple[str, ...]
    procedural: bool = False

    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)


@dataclass(frozen=True)
class TermMatch:
    """A synonym-group term found in a query."""

    term: str          # the matched surface form, lowercased
    group: SynonymGroup
    start: int         # token offset, for phrase-aware substitution
    length: int        # in tokens


@dataclass
class DomainKnowledge:
    """The loaded domain tables plus the lookups derived from them."""

    groups: tuple[SynonymGroup, ...]
    abbreviations: Mapping[str, str]
    acronyms: Mapping[str, str]
    spell_corrections: Mapping[str, str]
    intents: tuple[IntentRule, ...]
    source: str = "builtin"

    # --- derived, built once in __post_init__ ---
    _lookup: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)
    _group_by_term: dict[str, SynonymGroup] = field(default_factory=dict, repr=False)
    _phrases: tuple[tuple[str, ...], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        lookup: dict[str, set[str]] = {}
        group_by_term: dict[str, SynonymGroup] = {}
        phrases: set[tuple[str, ...]] = set()

        for group in self.groups:
            for term in group.terms:
                key = term.lower()
                # A term can sit in more than one group ("access" is a login
                # verb and a generic action); its expansion is the union, and
                # the FIRST group wins for entity attribution so the result is
                # deterministic rather than dependent on dict ordering.
                lookup.setdefault(key, set()).update(
                    other for other in group.terms if other.lower() != key
                )
                group_by_term.setdefault(key, group)
                tokens = tuple(key.replace("-", " ").split())
                if len(tokens) > 1:
                    phrases.add(tokens)

        self._lookup = {k: tuple(sorted(v)) for k, v in lookup.items()}
        self._group_by_term = group_by_term
        # Longest first: "learning management system" must be tried before
        # "system", or the short term swallows the phrase.
        self._phrases = tuple(sorted(phrases, key=len, reverse=True))

    # -- accessors ------------------------------------------------------

    @property
    def synonym_lookup(self) -> dict[str, tuple[str, ...]]:
        """Lowercased term → every other term in its group(s)."""
        return self._lookup

    @property
    def phrases(self) -> tuple[tuple[str, ...], ...]:
        """Multi-word terms as token tuples, longest first."""
        return self._phrases

    def group_for(self, term: str) -> Optional[SynonymGroup]:
        return self._group_by_term.get(term.lower())

    def synonyms_for(self, term: str) -> tuple[str, ...]:
        return self._lookup.get(term.lower(), ())

    @property
    def legacy_groups(self) -> list[list[str]]:
        """``[[term, ...], ...]`` — the shape the old module-level constant had.

        ``lexical.py`` builds BM25 alias lists from this. Kept so that module
        needs no change and so any external caller importing SYNONYM_GROUPS
        keeps working.
        """
        return [list(g.terms) for g in self.groups]

    # -- matching -------------------------------------------------------

    def match_terms(self, normalized: str) -> list[TermMatch]:
        """Every synonym-group term present in ``normalized``, phrases first.

        This is the single place phrase resolution happens. Synonym expansion
        and entity extraction both call it, so the two can never disagree about
        what the query mentions.

        Tokens consumed by a phrase are not offered to single-token matching,
        which is what stops "learning management system" also matching the
        "system" fragment of another group.
        """
        if not normalized:
            return []

        tokens = normalized.lower().split()
        consumed = [False] * len(tokens)
        matches: list[TermMatch] = []

        for phrase in self._phrases:
            n = len(phrase)
            if n > len(tokens):
                continue
            for i in range(len(tokens) - n + 1):
                if any(consumed[i:i + n]):
                    continue
                if tuple(tokens[i:i + n]) != phrase:
                    continue
                term = " ".join(phrase)
                group = self._group_by_term.get(term)
                if group is None:
                    continue
                matches.append(TermMatch(term=term, group=group, start=i, length=n))
                for j in range(i, i + n):
                    consumed[j] = True

        for i, token in enumerate(tokens):
            if consumed[i]:
                continue
            group = self._group_by_term.get(token)
            if group is not None:
                matches.append(TermMatch(term=token, group=group, start=i, length=1))

        matches.sort(key=lambda m: m.start)
        return matches

    def classify_intents(self, normalized: str) -> list[IntentRule]:
        """Every intent whose pattern matches, in declaration order.

        More than one can fire — "I forgot my password and can't log in" is
        genuinely both — and the caller decides how to combine them. Returning
        all of them rather than an argmax keeps that decision out of here.
        """
        if not normalized:
            return []
        return [rule for rule in self.intents if rule.matches(normalized)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _coerce_str_map(raw: Any, *, where: str) -> dict[str, str]:
    """YAML mapping → ``{lowercased str: str}``, skipping malformed entries.

    Keys are lowercased because every consumer looks up a lowercased token.
    A bad entry is logged and dropped rather than raising: one typo in an
    operator-edited file must not take retrieval down.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        logger.warning("domain_knowledge: %s must be a mapping, got %s — ignored",
                       where, type(raw).__name__)
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if key is None or value is None:
            continue
        out[str(key).strip().lower()] = str(value).strip()
    return out


def _parse_groups(raw: Any, defaults: dict[str, dict[str, Any]]) -> tuple[SynonymGroup, ...]:
    """Merge YAML synonym groups over the built-in ones, by group id."""
    merged: dict[str, dict[str, Any]] = {k: dict(v) for k, v in defaults.items()}

    if isinstance(raw, Mapping):
        for group_id, spec in raw.items():
            if not isinstance(spec, Mapping):
                logger.warning("domain_knowledge: synonym group %r is not a mapping — ignored",
                               group_id)
                continue
            terms = spec.get("terms") or []
            if not isinstance(terms, list) or not terms:
                logger.warning("domain_knowledge: synonym group %r has no terms — ignored",
                               group_id)
                continue
            merged[str(group_id)] = {
                "canonical": str(spec.get("canonical") or terms[0]),
                "kind": str(spec.get("kind") or "object"),
                "terms": [str(t) for t in terms if str(t).strip()],
            }
    elif raw is not None:
        logger.warning("domain_knowledge: `synonyms` must be a mapping — using defaults")

    groups: list[SynonymGroup] = []
    for group_id, spec in merged.items():
        terms = tuple(dict.fromkeys(str(t).strip() for t in spec["terms"] if str(t).strip()))
        if not terms:
            continue
        kind = spec.get("kind", "object")
        if kind not in ("system", "action", "object"):
            logger.warning("domain_knowledge: group %r has unknown kind %r — treating as object",
                           group_id, kind)
            kind = "object"
        groups.append(
            SynonymGroup(
                id=str(group_id),
                canonical=str(spec.get("canonical") or terms[0]),
                kind=kind,
                terms=terms,
            )
        )
    return tuple(groups)


def _parse_intents(raw: Any, defaults: dict[str, dict[str, Any]]) -> tuple[IntentRule, ...]:
    """Merge YAML intents over the built-in ones, by intent name."""
    merged: dict[str, dict[str, Any]] = {k: dict(v) for k, v in defaults.items()}

    if isinstance(raw, Mapping):
        for name, spec in raw.items():
            if not isinstance(spec, Mapping):
                logger.warning("domain_knowledge: intent %r is not a mapping — ignored", name)
                continue
            merged[str(name)] = {
                "patterns": spec.get("patterns") or [],
                "phrasings": spec.get("phrasings") or [],
                "boost": spec.get("boost") or [],
                "procedural": bool(spec.get("procedural", False)),
            }
    elif raw is not None:
        logger.warning("domain_knowledge: `intents` must be a mapping — using defaults")

    rules: list[IntentRule] = []
    for name, spec in merged.items():
        compiled: list[re.Pattern[str]] = []
        for pattern in spec.get("patterns") or []:
            try:
                compiled.append(re.compile(str(pattern), re.I))
            except re.error as exc:
                # A broken regex disables one intent, not the whole pipeline.
                logger.warning("domain_knowledge: intent %r has invalid pattern %r (%s) — skipped",
                               name, pattern, exc)
        if not compiled:
            continue
        rules.append(
            IntentRule(
                name=str(name),
                patterns=tuple(compiled),
                phrasings=tuple(str(p) for p in (spec.get("phrasings") or [])),
                boost=tuple(str(b) for b in (spec.get("boost") or [])),
                procedural=bool(spec.get("procedural", False)),
            )
        )
    return tuple(rules)


def _builtin() -> DomainKnowledge:
    return DomainKnowledge(
        groups=_parse_groups(None, _BUILTIN_SYNONYMS),
        abbreviations=dict(_BUILTIN_ABBREVIATIONS),
        acronyms=dict(_BUILTIN_ACRONYMS),
        spell_corrections=dict(_BUILTIN_SPELL_CORRECTIONS),
        intents=_parse_intents(None, _BUILTIN_INTENTS),
        source="builtin",
    )


def load_domain_knowledge(path: Optional[str | Path] = None) -> DomainKnowledge:
    """Load the domain tables, overlaying ``path`` onto the built-in defaults.

    Any failure — missing file, unparseable YAML, PyYAML not installed — falls
    back to the built-ins with a warning. Retrieval degrades to the behaviour it
    had before this file existed rather than failing to start, which is the
    right trade for a file an operator edits by hand.
    """
    if path is None:
        return _builtin()

    config_path = Path(path)
    if not config_path.is_file():
        logger.info("domain_knowledge: %s not found — using built-in defaults", config_path)
        return _builtin()

    try:
        import yaml  # noqa: PLC0415 — optional at runtime, see docstring
    except ImportError:
        logger.warning("domain_knowledge: PyYAML not installed — using built-in defaults")
        return _builtin()

    try:
        with config_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:  # OSError, yaml.YAMLError, decode errors — all recoverable
        logger.warning("domain_knowledge: failed to read %s (%s) — using built-in defaults",
                       config_path, exc)
        return _builtin()

    if not isinstance(raw, Mapping):
        logger.warning("domain_knowledge: %s is not a YAML mapping — using built-in defaults",
                       config_path)
        return _builtin()

    knowledge = DomainKnowledge(
        groups=_parse_groups(raw.get("synonyms"), _BUILTIN_SYNONYMS),
        abbreviations={**_BUILTIN_ABBREVIATIONS,
                       **_coerce_str_map(raw.get("abbreviations"), where="abbreviations")},
        acronyms={**_BUILTIN_ACRONYMS,
                  **_coerce_str_map(raw.get("acronyms"), where="acronyms")},
        spell_corrections={**_BUILTIN_SPELL_CORRECTIONS,
                           **_coerce_str_map(raw.get("spell_corrections"),
                                             where="spell_corrections")},
        intents=_parse_intents(raw.get("intents"), _BUILTIN_INTENTS),
        source=str(config_path),
    )
    logger.info(
        "domain_knowledge loaded from %s: %d synonym groups (%d entity), "
        "%d intents, %d spell corrections",
        config_path,
        len(knowledge.groups),
        sum(1 for g in knowledge.groups if g.is_entity),
        len(knowledge.intents),
        len(knowledge.spell_corrections),
    )
    return knowledge


@lru_cache(maxsize=1)
def get_domain_knowledge() -> DomainKnowledge:
    """Process-wide singleton, built once from the configured path."""
    from backend.app.config import get_settings  # local: avoids an import cycle

    return load_domain_knowledge(get_settings().domain_knowledge_path)


def reload_domain_knowledge() -> DomainKnowledge:
    """Re-read the YAML file and drop every lookup derived from it.

    Invalidates the caches in ``query_processing`` and ``lexical`` too: both
    hold tables derived from these groups, and leaving them stale would mean an
    edited synonym takes effect for the vector query but not for BM25.
    """
    get_domain_knowledge.cache_clear()
    knowledge = get_domain_knowledge()

    from backend.app.rag import lexical, query_processing

    query_processing.reset_caches()
    lexical.invalidate_lexicon()
    lexical.invalidate_lexical_index()
    return knowledge
