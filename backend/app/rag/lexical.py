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
        self._bm25: Optional[BM25Okapi] = None

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
        # Weight the title and category tokens into the chunk's document too:
        # a query mentioning "moodle" should match a chunk whose title says
        # Moodle even when the body only says "the LMS". BM25 is pure lexical,
        # so appending the title gives it the boost embedding-based search gets
        # for free.
        weighted = [
            f"{doc} {self._corpus_meta[cid].get('title', '')} "
            f"{self._corpus_meta[cid].get('category', '')}"
            for cid, doc in zip(self._corpus_ids, self._corpus_docs)
        ]
        self._bm25 = BM25Okapi([tokenize(d) for d in weighted])
        logger.info(
            "LexicalIndex rebuilt over %d chunks", len(self._corpus_ids)
        )

    def ensure_loaded(self) -> None:
        """Build the index on first use.

        Called from the query path, so a fresh process (or one whose cache was
        invalidated by a re-ingest) transparently rebuilds instead of silently
        returning no lexical hits.
        """
        if self._bm25 is None:
            self.rebuild()

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Return [(chunk_id, bm25_score), ...] best-first. Empty corpus → []."""
        if not query.strip():
            return []
        self.ensure_loaded()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
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

    def vocabulary(self) -> set[str]:
        """Distinct corpus tokens — the lexicon fuzzy rewriting checks against."""
        self.ensure_loaded()
        vocab: set[str] = set()
        for doc in self._corpus_docs:
            for word in tokenize(doc):
                if len(word) >= 3:
                    vocab.add(word)
        for meta in self._corpus_meta.values():
            for word in tokenize(meta.get("title", "") or ""):
                if len(word) >= 3:
                    vocab.add(word)
        return vocab


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
