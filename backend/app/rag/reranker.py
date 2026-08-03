"""Cross-encoder reranking.

The bi-encoder used for retrieval embeds the query and each chunk *separately*,
so it can only compare them in a shared vector space — good enough to pull
candidates out of thousands, but weak at fine-grained ordering. A cross-encoder
reads the query and chunk **together** in one forward pass and scores their
actual relevance, which reorders a shortlist far more accurately.

It is ~10-40ms per pair on CPU, so it runs on the MMR shortlist only (a dozen
chunks), never the full corpus. Model load is lazy and failure is non-fatal:
if the weights cannot be fetched the retriever keeps the cosine ordering rather
than erroring out on every query.
"""

import threading
from functools import lru_cache
from typing import Any, Optional

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Lazily-loaded cross-encoder that scores (query, passage) pairs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Optional[Any] = None
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True when the model is loaded, or not yet tried and not disabled."""
        if not self.settings.rerank_enabled or self._load_failed:
            return False
        return True

    def _ensure_model(self) -> Optional[Any]:
        """Load the model once. Returns None if unavailable."""
        if self._model is not None or self._load_failed:
            return self._model
        with self._lock:
            if self._model is not None or self._load_failed:
                return self._model
            try:
                from sentence_transformers import CrossEncoder  # lazy import

                logger.info("Loading cross-encoder: %s", self.settings.rerank_model)
                self._model = CrossEncoder(self.settings.rerank_model)
                logger.info("Cross-encoder ready")
            except Exception as exc:  # noqa: BLE001 — any load failure degrades gracefully
                self._load_failed = True
                logger.warning(
                    "Cross-encoder unavailable (%s: %s) — falling back to cosine "
                    "ordering. Set RERANK_ENABLED=false to silence this.",
                    type(exc).__name__,
                    exc,
                )
            return self._model

    def score(self, query: str, passages: list[str]) -> Optional[list[float]]:
        """Relevance score per passage, or None when reranking is unavailable.

        None (not an empty list, not zeros) is deliberate: the caller must be
        able to distinguish "the reranker says these are all irrelevant" from
        "the reranker did not run", and keep its existing order in the latter
        case.
        """
        if not self.settings.rerank_enabled or not passages:
            return None
        model = self._ensure_model()
        if model is None:
            return None
        try:
            scores = model.predict([(query, passage) for passage in passages])
            return [float(s) for s in scores]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder scoring failed (%s) — keeping prior order", exc)
            return None

    def warmup(self) -> None:
        """Pre-load the model so the first real query doesn't pay for it."""
        if self.settings.rerank_enabled:
            self._ensure_model()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
