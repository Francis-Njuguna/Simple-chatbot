"""Cross-encoder reranking.

The bi-encoder used for retrieval embeds the query and each chunk *separately*,
so it can only compare them in a shared vector space — good enough to pull
candidates out of thousands, but weak at fine-grained ordering. A cross-encoder
reads the query and chunk **together** in one forward pass and scores their
actual relevance, which reorders a shortlist far more accurately.

It runs on the fused shortlist only (a handful of chunks), never the full corpus.
Cost per pair is hardware-dependent. Measured *warm* on a 4-core CPU box with no
GPU, int8: 18-26ms per pair. So RERANK_SHORTLIST x RERANK_QUERY_FORMS is the
dominant term in retrieval latency, but a modest one — the shipped 8 x 2 = 16
pairs is ~0.4s.

Two numbers to distrust if you find them quoted elsewhere in this repo:

* "123-166ms per pair" was briefly written here and in config.py. It is wrong:
  it came from cold first calls that also paid graph setup. Discard the first
  few predict() calls before timing this model, or you will measure that again.
* "~271ms per pair", derived from production's 8.7s for 32 pairs. Also wrong,
  and more misleading than the first: that request was slow because the whole
  process was starved, not because a pair costs 271ms. In the same log, a stage
  that does nothing but concatenate strings took 1,848ms, one Postgres SELECT
  took 14,953ms, and the same warm model embedded one short query in 95ms and in
  4,452ms a minute apart. Rerank absorbs the blame because it is the largest
  block of CPU work in the request, not because it is the problem. Do NOT shrink
  the shortlist or swap in a smaller model to chase it. See RERANK_LATENCY.md.

Model load is lazy and failure is non-fatal: if the weights cannot be fetched the
retriever keeps the cosine ordering rather than erroring out on every query.
"""

import threading
from functools import lru_cache
from typing import Any, Optional

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Upper bound on pairs sent to one predict() call. The point of batching is to
# let RERANK_SHORTLIST x RERANK_QUERY_FORMS land in a single forward pass, which
# at the shipped 8 x 2 is 16 — well under this. The cap only exists so that
# raising those settings degrades into several sensible batches instead of one
# enormous allocation. Chunks are ~500 chars (~130 tokens), so 64 pairs is small.
_MAX_BATCH_PAIRS = 64


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

    @staticmethod
    def _quantize_int8(model: Any) -> bool:
        """Swap the cross-encoder's Linear layers for dynamic int8 in place.

        Reranking is ~84% of retrieval wall time (scripts/profile_pipeline.py),
        and the two ways to score *fewer* pairs both cost recall, so making each
        pair cheaper is the remaining lever. Measured 1.25x on this box
        (35/40 paired trials, sign-test z=+4.74) — see scripts/bench_quantization.py.

        The swap target matters and both obvious ones are wrong:

        * ``model.model = q`` — ``model`` is a property, so nn.Module.__setattr__
          registers ``q`` as a NEW child of the CrossEncoder Sequential. forward()
          then hands the raw HF module a features dict and dies on
          ``input_ids.size()``.
        * ``wrapper.auto_model = q`` — ``auto_model`` is a read-only property
          alias. Assignment creates a shadowing instance attribute while forward()
          keeps using the fp32 module, so scores come back BIT-IDENTICAL and the
          quantization silently does nothing.

        The module actually used by forward() is the registered child
        ``wrapper._modules["model"]``, so that is what gets replaced. Returns
        True only if the weights genuinely changed.
        """
        import torch
        from torch.quantization import quantize_dynamic

        try:
            wrapper = model[0]
        except (TypeError, IndexError, KeyError):
            logger.warning("Cross-encoder has unexpected layout — skipping int8")
            return False

        modules = getattr(wrapper, "_modules", {})
        inner = modules.get("model")
        if not isinstance(inner, torch.nn.Module):
            logger.warning(
                "Cross-encoder wrapper %s exposes no 'model' child (%s) — "
                "skipping int8",
                type(wrapper).__name__,
                list(modules),
            )
            return False

        quantized = quantize_dynamic(inner, {torch.nn.Linear}, dtype=torch.qint8)
        n_q = sum(
            1 for _, mod in quantized.named_modules()
            if "quantized" in type(mod).__module__
        )
        if n_q == 0:
            logger.warning("int8 quantization produced no quantized modules — skipping")
            return False

        modules["model"] = quantized
        logger.info("Cross-encoder quantized to int8 (%d quantized modules)", n_q)
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
                model = CrossEncoder(self.settings.rerank_model)
                if getattr(self.settings, "rerank_quantize", False):
                    # Failure here is non-fatal: an un-quantized reranker is
                    # slower but correct, which beats no reranker at all.
                    try:
                        self._quantize_int8(model)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "int8 quantization failed (%s: %s) — using fp32",
                            type(exc).__name__,
                            exc,
                        )
                self._model = model
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
            scores = model.predict(
                [(query, passage) for passage in passages],
                show_progress_bar=False,
            )
            return [float(s) for s in scores]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder scoring failed (%s) — keeping prior order", exc)
            return None

    def score_multi(
        self, queries: list[str], passages: list[str]
    ) -> Optional[list[float]]:
        """Best score per passage across several query phrasings, in ONE batch.

        Returns the same numbers as calling :meth:`score` once per query and
        max-pooling elementwise — identical pairs, identical model, identical
        arithmetic. What changes is scheduling, and on this hardware scheduling is
        worth real time:

        * One ``predict`` call instead of N means one batching pass. With
          shortlist 8 x 2 forms = 16 pairs and predict's default batch_size of
          32, the whole grid is a single forward batch where it used to be two.
        * ``predict`` length-sorts its input to minimise padding
          (``CrossEncoder.predict``, and it un-sorts the results afterwards so
          input order is preserved — which is what makes the reshape below
          correct). Sorting all pairs *together* pads better than sorting each
          form's batch separately.

        Falls back to None on any failure, exactly like :meth:`score`, so the
        caller keeps its prior ordering rather than erroring out.
        """
        if not self.settings.rerank_enabled or not passages or not queries:
            return None
        model = self._ensure_model()
        if model is None:
            return None

        # Row-major: all passages for queries[0], then all for queries[1], ...
        pairs = [(q, p) for q in queries for p in passages]
        try:
            # show_progress_bar defaults to None, which sentence-transformers
            # resolves to True whenever its logger sits at INFO — that is a tqdm
            # bar on stderr for every query in production. Always off here.
            flat = model.predict(
                pairs,
                batch_size=min(len(pairs), _MAX_BATCH_PAIRS),
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder scoring failed (%s) — keeping prior order", exc)
            return None

        if len(flat) != len(pairs):  # defensive: never seen, but the reshape assumes it
            logger.warning(
                "Cross-encoder returned %d scores for %d pairs — keeping prior order",
                len(flat),
                len(pairs),
            )
            return None

        width = len(passages)
        best = [float(s) for s in flat[:width]]
        for offset in range(width, len(pairs), width):
            for i in range(width):
                value = float(flat[offset + i])
                if value > best[i]:
                    best[i] = value
        return best

    def warmup(self) -> None:
        """Pre-load the model so the first real query doesn't pay for it."""
        if self.settings.rerank_enabled:
            self._ensure_model()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
