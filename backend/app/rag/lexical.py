"""Lexical retrieval: BM25 index + domain fuzzy query rewriting.

Two roles here, both local and cheap:

* :class:`LexicalIndex` builds a BM25 index over the text corpus once per
  process and answers ranked lexical queries. The vector index is very good at
  paraphrase but weak on exact tokens ("SMOWL", "VAS", "moodle"); BM25 catches
  exactly those. It is read-only, rebuilt on re-ingest (the corpus changed) and
  cached with ``@lru_cache`` so the hot path never rescans Chroma.
* :func:`rewrite_query` fuzzy-corrects domain terms in the user's question —
  "smwol" / "smowl" → "SMOWL", "athenticator" → "authenticator" — against the
  KB's own vocabulary (titles + chunk tokens), using difflib's SequenceMatcher.
  This is deliberate: an acronym embedded in a typo is invisible to both BM25
  (no token match) and a bi-encoder (sub-word split), but a one-token fuzzy
  fix brings the exact token back. No LLM, no network, well under a millisecond.
"""

import difflib
import re
from functools import lru_cache
from typing import Optional

from rank_bm25 import BM25Okapi

from backend.app.config import get_settings
from backend.app.database.chroma import fetch_all_text_documents
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Tokeniser: lowercase, keep letters/digits/apostrophes/hyphens (so "student's"
# and "re-register" survive as units), drop everything else. BM25 scores are
# built on exact token hits, so this must match the spelling of what users type.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[''-][a-z0-9]+)*", re.IGNORECASE)

# Field weights for BM25 indexing. Metadata fields are repeated N times so term
# frequency reflects their importance: a title match outweighs an incidental
# body mention. Weights are deliberately small — a title should boost, not
# dominate a chunk that genuinely answers the question.
#
# ``title`` is not a Chroma metadata key: the ingest pipeline deliberately drops
# title/url from Chroma (PostgreSQL owns them, see ``metadata_service``). It is
# recovered from the ``[Title]`` header the chunker prepends to every chunk
# body, which is verified present on all of them. Reading it from the text keeps
# this index buildable from Chroma alone — it is constructed synchronously and
# lazily on the query path, where no AsyncSession is available.
_FIELD_WEIGHTS = {
    "title": 3,
    "category": 2,
    "summary": 1,
    "keywords": 2,
}

# The ``[Title]`` header the chunker prepends to every chunk body.
_TITLE_HEADER_RE = re.compile(r"^\s*\[([^\]]{1,200})\]")


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, stopwords excluded."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# Common English stopwords. BM25 Okapi's IDF handles function words fine, but
# dropping them makes the index tighter and query scoring less noisy.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my",
    "not", "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "then", "there", "they", "this", "to", "up", "us", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "you", "your",
}


class LexicalIndex:
    """BM25 index over the Chroma text corpus, rebuilt when it changes."""

    def __init__(self) -> None:
        self._corpus_ids: list[str] = []
        self._corpus_docs: list[str] = []
        self._corpus_meta: dict[str, dict] = {}
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._vocabulary: Optional[frozenset[str]] = None

    @property
    def loaded(self) -> bool:
        return self._bm25 is not None

    def rebuild(self) -> None:
        data = fetch_all_text_documents()
        ids = data["ids"]
        documents = data["documents"]
        metadatas = data["metadatas"]
        if not ids:
            logger.warning("LexicalIndex.rebuild: no text documents in Chroma")
            self._bm25 = None
            return

        self._corpus_ids = ids
        self._corpus_docs = [doc or "" for doc in documents]
        self._corpus_meta = {
            cid: meta or {} for cid, meta in zip(ids, metadatas)
        }

        # Index the article's *identity* alongside its body.
        #
        # A chunk's body often never names the system it documents — the LMS
        # login article says "enter your credentials", not "Moodle" — so a
        # query naming the system has zero lexical overlap with the very chunk
        # that answers it. Title, category, summary and keywords carry those
        # names, so folding them in is what lets BM25 match "Moodle login"
        # against a body that only says "the portal".
        #
        # Fields are repeated ``_FIELD_WEIGHTS`` times rather than appended
        # once. BM25 scores on term frequency, so repetition is how a pure-
        # lexical index expresses "a title hit means more than a body hit"
        # without modifying the scorer. The weights are deliberately small:
        # a title match should outrank an incidental body mention, not
        # dominate a chunk that genuinely answers the question.
        weighted: list[str] = []
        for cid, doc in zip(self._corpus_ids, self._corpus_docs):
            meta = self._corpus_meta[cid]
            parts = [doc]
            # Title comes from the chunk's own [Title] header, not metadata —
            # Chroma does not carry it (see _FIELD_WEIGHTS).
            title_match = _TITLE_HEADER_RE.match(doc)
            fields = dict(meta)
            if title_match:
                fields["title"] = title_match.group(1)
            for field, weight in _FIELD_WEIGHTS.items():
                value = str(fields.get(field) or "").strip()
                if value:
                    parts.extend([value] * weight)
            # Synonyms of what the title names, so "LMS login" reaches the
            # chunk whose title says Moodle. Added once (weight 1): an alias
            # is weaker evidence than the article's own words.
            aliases = _aliases_for(
                " ".join(
                    str(fields.get(f) or "") for f in ("title", "category", "keywords")
                )
            )
            if aliases:
                parts.append(" ".join(aliases))
            weighted.append(" ".join(parts))

        self._corpus_tokens = [tokenize(d) for d in weighted]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._vocabulary = None  # invalidated with the corpus
        logger.info(
            "LexicalIndex rebuilt over %d chunks (fields=%s)",
            len(self._corpus_ids),
            ",".join(_FIELD_WEIGHTS),
        )

    def ensure_loaded(self) -> None:
        """Build the index on first use.

        Called from the query path, so a fresh process (or one whose cache was
        invalidated by a re-ingest) transparently rebuilds instead of silently
        returning no lexical hits.
        """
        if self._bm25 is None:
            self.rebuild()

    def search(
        self, query: str, k: int = 20, fuzzy: bool = True
    ) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score), ...] best-first. Empty corpus → [].

        With ``fuzzy=True``, a query token absent from the corpus vocabulary is
        replaced by its closest in-vocabulary neighbour before scoring. This is
        the last line of typo defence: the explicit correction map in
        :func:`rewrite_query` covers known misspellings, but a novel one
        ("registartion") would otherwise contribute nothing to a BM25 query
        whose entire mechanism is exact token match — one typo silently drops
        the term from the query.

        Matching against the *corpus* vocabulary rather than a dictionary is
        what makes this safe: the only substitutions available are words this
        knowledge base actually uses.
        """
        if not query.strip():
            return []
        self.ensure_loaded()
        if self._bm25 is None:
            return []

        tokens = tokenize(query)
        if fuzzy:
            tokens = self._fuzzy_tokens(tokens)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        # Skip zero-scoring tail — no lexical overlap at all.
        results: list[tuple[str, float]] = []
        for idx in ranked:
            if scores[idx] <= 0.0:
                continue
            results.append((self._corpus_ids[idx], float(scores[idx])))
            if len(results) >= k:
                break
        return results

    def _fuzzy_tokens(self, tokens: list[str]) -> list[str]:
        """Map out-of-vocabulary tokens to their nearest corpus term.

        Short tokens (< 4 chars) are left alone: at that length a single edit
        reaches too many unrelated words for the match to mean anything.
        """
        vocab = self.vocabulary()
        if not vocab:
            return tokens

        out: list[str] = []
        for token in tokens:
            if token in vocab or len(token) < 4:
                out.append(token)
                continue
            match = difflib.get_close_matches(token, vocab, n=1, cutoff=0.82)
            if match:
                logger.debug("BM25 fuzzy token: %r → %r", token, match[0])
                # Keep both: the original may be a real word missing from this
                # corpus, and dropping it would lose a genuine constraint.
                out.extend([token, match[0]])
            else:
                out.append(token)
        return out

    def vocabulary(self) -> frozenset[str]:
        """Distinct tokens across everything indexed — bodies and metadata.

        Cached with the corpus: this is called once per query on the fuzzy path
        and recomputing it over every chunk each time would dominate BM25's
        own cost.
        """
        self.ensure_loaded()
        if self._vocabulary is not None:
            return self._vocabulary
        vocab: set[str] = set()
        # _corpus_tokens is the *weighted* document set, so titles, categories,
        # summaries, keywords and aliases are all already present.
        for tokens in self._corpus_tokens:
            for word in tokens:
                if len(word) >= 3:
                    vocab.add(word)
        self._vocabulary = frozenset(vocab)
        return self._vocabulary


@lru_cache(maxsize=1)
def get_lexical_index() -> LexicalIndex:
    return LexicalIndex()


def invalidate_lexical_index() -> None:
    """Drop the cached index so the next search rebuilds from Chroma.

    Called by the ingest pipeline after a successful re-ingest.
    """
    get_lexical_index.cache_clear()
    logger.info("Lexical index cache cleared (will rebuild on next query)")


# ---------------------------------------------------------------------------
# Domain fuzzy rewrite
# ---------------------------------------------------------------------------

# Terms that must never be mangled by fuzzy correction: the acronyms and
# product names the KB cares about. All-lowercase keys; compare casefolded.
_DOMAIN_VOCAB = {
    "amiu", "lms", "moodle", "smowl", "swoml", "swowl", "vas", "mfa",
    "authenticator", "password", "portal", "student", "exams", "proctoring",
    "assignments", "campus", "email", "microsoft", "m365", "teams", "grades",
    "knowledgebase", "login", "register", "course", "transcript", "fee",
}

# Exact misspellings observed in real queries (or the obvious next typo), mapped
# to the canonical form. Fuzzy matching alone can't reach these: "smwol" is a
# letter transposition (SequenceMatcher ratio 0.80) and "moddle" is one edit
# (ratio 0.83), both below the conservative fuzzy bar — yet both are exactly
# what a user types when hunting for "SMOWL"/"Moodle".
_DOMAIN_CORRECTIONS = {
    "smwol": "SMOWL",
    "swoml": "SMOWL",
    "swowl": "SMOWL",
    "athenticator": "authenticator",
    "authentificator": "authenticator",
    "moddle": "Moodle",
    "moodle": "Moodle",
    "pasword": "password",
    "paswword": "password",
}


def rewrite_query(query: str) -> str:
    """Fuzzy-correct misspelled domain tokens in ``query``, in place.

    Strategy, per token:
    1. If it's an exact known term, leave it.
    2. If it's in the curated correction map, use the canonical form
       ("smwol" → "SMOWL").
    3. Otherwise, if it closely matches a known term (ratio ≥ 0.86), use that.

    A token is only rewritten when the match is unambiguous, so a normal word is
    never "corrected" into something else.

    Returns the rewritten query; the original when nothing changed.
    """
    if not query.strip():
        return query

    tokens = query.split()
    lexicon = _lexicon()
    rewritten: list[str] = []
    changed = False

    for token in tokens:
        cleaned = re.sub(r"[^a-z0-9]", "", token, flags=re.IGNORECASE).lower()
        if len(cleaned) < 3:
            rewritten.append(token)
            continue

        if cleaned in _DOMAIN_CORRECTIONS:
            rewritten.append(_DOMAIN_CORRECTIONS[cleaned])
            changed = True
            continue
        if cleaned in lexicon:
            rewritten.append(token)
            continue

        best, best_ratio = "", 0.0
        for candidate in _DOMAIN_VOCAB:
            ratio = difflib.SequenceMatcher(None, cleaned, candidate).ratio()
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio
        # High bar: only unambiguous near-matches. 0.86 ≈ 1 edit in ~7 chars,
        # which is the "smwol"→"smowl" class of typo and not much more.
        if best and best_ratio >= 0.86:
            replacement = best.upper() if best in {"lms", "smowl", "vas", "mfa", "m365"} else best
            rewritten.append(replacement)
            changed = True
        else:
            rewritten.append(token)

    if not changed:
        return query
    result = " ".join(rewritten)
    logger.info("Query rewrite: %r -> %r", query, result)
    return result


@lru_cache(maxsize=1)
def _lexicon() -> frozenset[str]:
    """Tokens known to the KB — titles plus the index corpus."""
    return frozenset(_DOMAIN_VOCAB | get_lexical_index().vocabulary())


def invalidate_lexicon() -> None:
    """Clear the cached lexicon after a re-ingest (titles may have changed)."""
    _lexicon.cache_clear()


def _aliases_for(text: str) -> list[str]:
    """Domain synonyms of terms in ``text``, for title/category indexing.

    Returns a flat list of aliases so "LMS login" in a title contributes
    "Moodle", "Learning Management System", etc. to the chunk's BM25 document.
    """
    # Import here to avoid circular dependency — query_processing imports config,
    # which may indirectly import this module during app init.
    from backend.app.rag.query_processing import SYNONYM_GROUPS

    aliases: set[str] = set()
    tokens = set(tokenize(text))
    for group in SYNONYM_GROUPS:
        group_lower = [t.lower() for t in group]
        if any(tok in group_lower for tok in tokens):
            aliases.update(group)
    # Remove terms that are already in the input text.
    return [a for a in aliases if a.lower() not in tokens]
