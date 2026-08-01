"""Extractive article summarisation.

Runs during ingestion, so it must be fast, dependency-free and deterministic —
no LLM call, no network. The optional abstractive summary is a separate
background pass (see ``services/enrichment_service.py``).

Approach: a small TextRank-flavoured scorer.

1. Split the article into sentences.
2. Score each sentence by term overlap with the whole document, using
   log-scaled term frequencies so a repeated word stops dominating.
3. Boost early sentences (help-desk articles state the problem up front) and
   procedural lines ("click", "select", "navigate") since those carry the
   actionable content the RAG prompt benefits from most.
4. Emit the top sentences **in original document order** so the summary reads
   as prose rather than a ranked list.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Words too common to signal what a sentence is about.
_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself me more most my myself no nor
    not of off on once only or other ought our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours yourself yourselves
    """.split()
)

# Verbs that mark an actionable instruction — the most useful thing to keep.
_PROCEDURAL_CUES = frozenset(
    """
    click select choose enter type navigate open close login logout sign log press
    tap scan install download upload configure set reset submit confirm verify
    complete follow go visit check ensure contact request
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[a-z0-9']+")

# Bounds that keep a "sentence" a real sentence: shorter is a heading or menu
# fragment, longer is usually unsplit boilerplate.
_MIN_SENTENCE_CHARS = 25
_MAX_SENTENCE_CHARS = 400


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text or ""):
        sentence = " ".join(raw.split())
        if _MIN_SENTENCE_CHARS <= len(sentence) <= _MAX_SENTENCE_CHARS:
            sentences.append(sentence)
    return sentences


def _tokenize(sentence: str) -> list[str]:
    return [w for w in _WORD.findall(sentence.lower()) if w not in _STOPWORDS and len(w) > 2]


def summarize_extractive(
    text: str,
    *,
    max_sentences: int = 5,
    max_chars: int = 1200,
    title: str | None = None,
) -> str:
    """Return an extractive summary of ``text`` (empty string when unusable).

    ``title`` — when given, sentences overlapping the title score higher; an
    article's title is the best available statement of its topic.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return _join_within(sentences, max_chars)

    # Document-level term frequency, log-scaled so repetition saturates.
    doc_terms = Counter()
    tokenized: list[list[str]] = []
    for sentence in sentences:
        tokens = _tokenize(sentence)
        tokenized.append(tokens)
        doc_terms.update(tokens)
    weights = {term: 1.0 + math.log(count) for term, count in doc_terms.items()}

    title_terms = set(_tokenize(title)) if title else set()
    total = len(sentences)

    scored: list[tuple[float, int]] = []
    for index, tokens in enumerate(tokenized):
        if not tokens:
            continue
        # Mean term weight — mean, not sum, so long sentences don't win by length.
        score = sum(weights.get(t, 0.0) for t in tokens) / len(tokens)
        # Position: earliest sentences carry the problem statement.
        score *= 1.0 + 0.5 * (1.0 - index / total)
        if title_terms and title_terms.intersection(tokens):
            score *= 1.15
        if _PROCEDURAL_CUES.intersection(tokens):
            score *= 1.2
        scored.append((score, index))

    if not scored:
        return _join_within(sentences, max_chars)

    scored.sort(key=lambda pair: pair[0], reverse=True)
    chosen = sorted(index for _, index in scored[:max_sentences])
    return _join_within([sentences[i] for i in chosen], max_chars)


def _join_within(sentences: list[str], max_chars: int) -> str:
    """Join sentences, stopping at ``max_chars`` on a sentence boundary."""
    out: list[str] = []
    length = 0
    for sentence in sentences:
        addition = len(sentence) + (1 if out else 0)
        if length + addition > max_chars:
            break
        out.append(sentence)
        length += addition
    if not out and sentences:
        # A single oversized sentence — hard-truncate on a word boundary.
        return sentences[0][:max_chars].rsplit(" ", 1)[0].strip()
    return " ".join(out)
