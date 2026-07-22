"""ChromaDB client and collection management.

Client selection
----------------
``get_chroma_client()`` picks the backend based on config (see Settings):

- **HTTP mode** — connects to a standalone Chroma server (e.g. a separate
  Railway service) via ``chromadb.HttpClient``. Chosen when
  ``CHROMA_MODE=http`` or (``CHROMA_MODE=auto`` and ``CHROMA_SERVER_HOST``
  is set).
- **Persistent mode** — reads an on-disk store via ``chromadb.PersistentClient``
  at ``CHROMA_PERSIST_DIR``. Used for the pre-built vector store baked into the
  Docker image, or a mounted Railway volume. This is the default.

Performance notes
-----------------
* Both the client *and* the collection handles are cached (``lru_cache``) so we
  never re-open the on-disk store or re-issue ``get_or_create_collection`` on
  the hot query path.
* ``query_text_collection`` now optionally returns the stored embeddings
  (``include_embeddings=True``) so callers can run MMR / reranking without
  re-embedding candidate chunks over the network.
"""

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

TEXT_COLLECTION = "amref_text_chunks"
IMAGE_COLLECTION = "amref_image_embeddings"


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()

    if settings.use_chroma_http:
        host = settings.chroma_server_host or settings.chroma_host
        port = settings.chroma_server_port
        client = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=settings.chroma_server_ssl,
        )
        logger.info(
            "ChromaDB HttpClient connected → %s://%s:%d",
            "https" if settings.chroma_server_ssl else "http",
            host,
            port,
        )
        return client

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    logger.info("ChromaDB PersistentClient initialized at %s", settings.chroma_persist_dir)
    return client


@lru_cache
def get_text_collection() -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache
def get_image_collection() -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _reset_collection_cache() -> None:
    """Drop cached collection handles (needed after delete/recreate)."""
    get_text_collection.cache_clear()
    get_image_collection.cache_clear()


def upsert_text_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    collection = get_text_collection()
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def upsert_image_embeddings(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    collection = get_image_collection()
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_text_collection(
    query_embedding: list[float],
    n_results: int = 10,
    where: dict[str, Any] | None = None,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    collection = get_text_collection()
    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=include,
    )


def query_image_collection(
    query_embedding: list[float],
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    collection = get_image_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def clear_collections() -> None:
    client = get_chroma_client()
    for name in [TEXT_COLLECTION, IMAGE_COLLECTION]:
        try:
            client.delete_collection(name)
        except ValueError:
            pass
    _reset_collection_cache()
    get_text_collection()
    get_image_collection()


def check_embedding_dimension(expected_dim: int | None, log_only: bool = True) -> bool | None:
    """Inspect one stored embedding in the text collection and compare its length.

    - expected_dim: the configured embedding_dim from Settings (or None if unknown)
    - log_only: when False, raise RuntimeError on mismatch; when True, only log.

    Returns:
      - True  => detected and matches expected_dim
      - False => detected and DOES NOT match expected_dim
      - None  => could not determine (no embeddings or inspection failed)

    This is a best-effort check: Chroma collection APIs vary across versions, so
    this helper attempts a couple of common ways to read stored embeddings and
    falls back gracefully if unsupported.
    """
    collection = get_text_collection()

    try:
        # Preferred: read a small sample without running a query.
        try:
            # Many chroma versions support collection.get(include=["embeddings"], limit=1)
            sample = collection.get(include=["embeddings"], limit=1)
        except TypeError:
            # Some clients omit 'limit' — try without it.
            sample = collection.get(include=["embeddings"])
    except Exception:
        # Fallback: use a cheap query to return one item with embeddings.
        try:
            sample = collection.query(query_embeddings=[[0.0]], n_results=1, include=["embeddings"])  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma embedding-dim check failed to read sample embedding: %s", exc)
            return None

    # Normalize the returned shape to a list-of-lists pattern used elsewhere.
    embeddings = sample.get("embeddings", [[]]) if isinstance(sample, dict) else None
    if embeddings is None:
        logger.warning("Chroma embedding-dim check: collection returned unexpected payload.")
        return None

    # embeddings may be [[...]] or []
    try:
        first = embeddings[0]
    except Exception:
        logger.warning("Chroma embedding-dim check: no embeddings found in collection sample.")
        return None

    if not isinstance(first, list):
        # Sometimes embeddings may be returned as numpy arrays or other types — coerce if possible
        try:
            first = list(first)
        except Exception:
            logger.warning("Chroma embedding-dim check: could not coerce sample embedding to list.")
            return None

    actual_dim = len(first)
    if expected_dim is None:
        logger.info("Chroma embedding-dim detected: %d (no expected dim configured)", actual_dim)
        return None

    if actual_dim != expected_dim:
        msg = (
            f"CRITICAL: Chroma embedding dimension mismatch — store={actual_dim} vs "
            f"configured={expected_dim}. This likely means the vector DB was built with a "
            "different embedding model. Set EMBEDDING_PROVIDER/EMBEDDING_MODEL to match "
            "the stored vectors, or re-ingest the vector store with the desired model."
        )
        if log_only:
            logger.critical(msg)
            return False
        raise RuntimeError(msg)

    logger.info("Chroma embedding dimension OK — %d", actual_dim)
    return True
