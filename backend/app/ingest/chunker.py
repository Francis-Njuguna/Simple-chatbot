"""Structure-aware text chunking for ingestion.

Two properties matter for retrieval quality here, and a plain character splitter
gives neither:

**1. Chunks should break at semantic boundaries.**
``RecursiveCharacterTextSplitter`` cuts at a fixed character count, so a
numbered procedure routinely loses its final steps to the next chunk — the
retriever then returns "STEP 1-4" while the answer the user needed was step 5.
We instead accumulate whole paragraphs (blank-line separated, which is exactly
what the block-aware HTML cleaner emits) and only start a new chunk when adding
the next paragraph would exceed the budget. Paragraphs longer than the budget on
their own fall back to sentence-level packing, and only a single sentence longer
than the budget is ever hard-split.

**2. Every chunk should carry its topic.**
Article bodies use pronouns and bare nouns — "click Assignments", "the LMS" —
so chunk 3 of the Moodle guide can contain no token identifying it as Moodle.
The embedding then sits far from a "Moodle assignments" query, and BM25 has
nothing to match at all. Prefixing each chunk with its title and category (a
"contextual chunk header") puts the topic into both the vector and the lexical
index. It costs a handful of tokens per chunk and closes exactly the
vocabulary gap that made "how do I submit an assignment in Moodle?" retrieve
nothing useful.

The header is part of the stored document deliberately: it is what Chroma
embeds, what BM25 tokenises, and what the LLM reads as context — all three
benefit from knowing which article the text came from.
"""

import re
from typing import Any

from backend.app.config import get_settings

# Paragraph boundary: one or more blank lines. The HTML cleaner (utils/text.py)
# emits newlines at block boundaries, so this tracks the document's real
# structure rather than guessing from prose.
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")

# Sentence boundary for over-long paragraphs. Requires the following character to
# be upper-case/digit so "Step 1. Click" and "amref.ac.ke" don't split.
_SENTENCE_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9])")

# A line that introduces a section rather than being prose: "STEP 2: ...",
# "RECOMMENDED BROWSERS", "How to reset your password". Starting a new chunk at
# one of these keeps a procedure's heading attached to its steps.
_HEADING_RE = re.compile(
    r"^(?:step\s*\d+\b|[A-Z][A-Z\s/&-]{4,}$|(?:how|what|why|where|when)\b.{0,60}\?$)",
    re.IGNORECASE,
)


class TextChunker:
    """Splits article text into retrieval-sized, topic-labelled chunks."""

    def __init__(self) -> None:
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    # -- public API ---------------------------------------------------------

    def chunk_article(
        self,
        article_id: str,
        title: str,
        category: str | None,
        url: str,
        text: str,
    ) -> list[dict[str, Any]]:
        if not text.strip():
            return []

        header = self._build_header(title, category)
        # The header is prepended to every chunk, so it has to come out of the
        # body's budget or chunks overrun the embedding model's window.
        body_budget = max(self.chunk_size - len(header), 200)

        pieces = self._split_text(text, body_budget)

        result: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(pieces):
            result.append(
                {
                    "chunk_id": f"{article_id}_chunk_{idx}",
                    "text": f"{header}{chunk_text}",
                    "metadata": {
                        "article_id": article_id,
                        "title": title,
                        "category": category or "General",
                        "url": url,
                        "chunk_index": idx,
                    },
                }
            )
        return result

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _build_header(title: str, category: str | None) -> str:
        """The contextual header prefixed to every chunk of this article."""
        title = (title or "").strip()
        category = (category or "").strip()
        if title and category and category.lower() not in title.lower():
            return f"[{category} — {title}]\n"
        if title:
            return f"[{title}]\n"
        if category:
            return f"[{category}]\n"
        return ""

    def _split_text(self, text: str, budget: int) -> list[str]:
        """Pack paragraphs into ``budget``-sized chunks at semantic boundaries."""
        paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if current:
                chunks.append("\n\n".join(current).strip())
                current, current_len = [], 0

        for para in paragraphs:
            # A paragraph too big to ever fit: emit what we have, then pack it
            # by sentence so we still break at a readable boundary.
            if len(para) > budget:
                flush()
                chunks.extend(self._split_paragraph(para, budget))
                continue

            # +2 for the "\n\n" join. Start a new chunk when the budget is spent,
            # or when this paragraph is a heading and we already have content —
            # a heading belongs with what follows it, not with what precedes it.
            starts_section = bool(_HEADING_RE.match(para.splitlines()[0].strip()))
            if current and (
                current_len + len(para) + 2 > budget
                or (starts_section and current_len > budget // 2)
            ):
                flush()

            current.append(para)
            current_len += len(para) + 2

        flush()
        return self._apply_overlap(chunks)

    def _split_paragraph(self, para: str, budget: int) -> list[str]:
        """Pack one over-long paragraph by sentence, hard-splitting only if forced."""
        sentences = _SENTENCE_RE.split(para)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            if len(sentence) > budget:
                if current:
                    chunks.append(" ".join(current).strip())
                    current, current_len = [], 0
                # Single sentence longer than a whole chunk — nothing semantic
                # left to break on, so slice it.
                for i in range(0, len(sentence), budget):
                    chunks.append(sentence[i : i + budget].strip())
                continue

            if current and current_len + len(sentence) + 1 > budget:
                chunks.append(" ".join(current).strip())
                current, current_len = [], 0

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunks.append(" ".join(current).strip())
        return [c for c in chunks if c]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Prepend the tail of each chunk to the next one.

        Overlap protects against a query whose answer straddles a boundary. We
        carry back whole sentences rather than a fixed character count, so the
        overlap never begins mid-word.
        """
        if self.chunk_overlap <= 0 or len(chunks) < 2:
            return chunks

        result = [chunks[0]]
        for previous, chunk in zip(chunks, chunks[1:]):
            tail = self._tail(previous, self.chunk_overlap)
            result.append(f"{tail}\n\n{chunk}" if tail else chunk)
        return result

    @staticmethod
    def _tail(text: str, max_chars: int) -> str:
        """Last whole sentence(s) of ``text``, up to ``max_chars``."""
        if len(text) <= max_chars:
            return text
        window = text[-max_chars:]
        # Drop a leading partial sentence so the overlap reads cleanly.
        match = _SENTENCE_RE.search(window)
        return window[match.end() :].strip() if match else window.strip()
