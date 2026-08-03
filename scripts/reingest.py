"""Force a full re-ingest: crawl → chunk → embed → store.

Mirrors what POST /ingest does, but runs standalone so no server/JWT is
needed. Uses the shared EmbeddingService (model loaded once) and the same
IngestionPipeline the API would construct.

Import ordering — please do not "tidy" this
-------------------------------------------
The heavy imports live inside ``main()`` on purpose, and the SentenceTransformer
is loaded *before* anything pulls in chromadb.

Loading torch in a process where chromadb (and its onnxruntime) had already
initialised killed this script with a Windows access violation — exit
-1073741819 / 0xC0000005, a native crash with no Python traceback, right at
"Loading sentence-transformer model". Probing the reverse order (torch first,
then Chroma, then even a second torch model for the cross-encoder) survived
every time.

Root cause: this venv contains TWO OpenMP runtimes —

    .venv/Lib/site-packages/torch/lib/libiomp5md.dll   (Intel, via torch)
    .venv/Lib/site-packages/sklearn/.libs/vcomp140.dll (Microsoft, via sklearn)

Both cannot safely initialise in one process: whichever loads second sets up its
thread pool over the first one's state and faults. Two mitigations are applied,
because either alone is fragile:

1. ``KMP_DUPLICATE_LIB_OK=TRUE`` (set at the top of this module, before any
   import that pulls in torch — the Intel runtime reads it at DLL init, so
   setting it afterwards does nothing).
2. Import ordering: load the SentenceTransformer *first*, so the Intel runtime
   wins the race and onnxruntime attaches to an already-initialised pool.

The FastAPI app already gets ordering right: ``_warmup()`` in
``backend/app/main.py`` loads the embedding model first and only then primes the
Chroma collections. This script now matches that order.

The FastAPI app already gets this right: ``_warmup()`` in ``backend/app/main.py``
loads the embedding model first and only then primes the Chroma collections.
This script now matches that order.
"""

import asyncio
import logging
import os
import sys

# Must be set BEFORE torch is imported — the Intel OpenMP runtime reads it at
# DLL init, so setting it later has no effect. See the module docstring.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, ".")


def _setup_logging() -> None:
    """Enable INFO logs.

    The app configures logging in its lifespan; this script has no lifespan, so
    without this every ``logger.info`` in the pipeline is dropped — which is how
    an earlier crash managed to look like it produced no output at all.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def main() -> None:
    _setup_logging()

    from backend.app.config import get_settings

    settings = get_settings()
    print(
        f"LLM_PROVIDER={settings.llm_provider}  "
        f"EMBEDDING_PROVIDER={settings.embedding_provider}  "
        f"EMBEDDING_MODEL={settings.embedding_model}",
        flush=True,
    )

    # 1. Embedding model FIRST — see the module docstring on import ordering.
    from backend.app.rag.embeddings import EmbeddingService

    embedder = EmbeddingService()
    embedder.embed_texts(["warmup"])  # force the full load + one forward pass
    print("Embedding model loaded.", flush=True)

    # 2. Only now touch Chroma.
    from backend.app.database.chroma import count_by_source_type

    print("Before:", count_by_source_type(), flush=True)

    from backend.app.database.session import async_session_factory
    from backend.app.ingest.pipeline import IngestionPipeline

    async with async_session_factory() as db:
        # Reuse the already-loaded embedder rather than constructing a second one.
        pipeline = IngestionPipeline(db, embedder)
        result = await pipeline.run(force=True, include_images=True)

    # pipeline.run() invalidates the metadata / BM25 / lexicon / retrieval caches
    # itself, so there is nothing to flush here.
    print("After:", count_by_source_type(), flush=True)
    print("Result:", result["status"], "|", result["message"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
