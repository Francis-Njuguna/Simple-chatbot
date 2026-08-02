"""ChromaDB client and collection management.

Collection layout
-----------------
A **single** collection (``amref_knowledge``) holds both text chunks and image
embeddings, discriminated by the ``source_type`` metadata field (``"text"`` or
``"image"``). One collection means one HNSW index to warm, one place to filter,
and image vectors that live in the same space as text vectors — an image is
embedded from its caption / alt text / nearby context, so a text query matches
it directly.

Metadata policy
---------------
PostgreSQL is the source of truth for metadata. Chroma stores only what is
needed to *retrieve* without a database round-trip:

    text  → source_type, article_id, chunk_id, chunk_index, category
    image → source_type, image_id, article_id, category

Everything else (title, url, caption, filename, static_path, keywords…) is
fetched from Postgres by ``article_id`` / ``image_id`` after retrieval. Do not
add display-only fields back here.

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
* Both the client *and* the collection handles are cached (behind a lock, since
  retrieval touches Chroma from worker threads) so we never re-open the on-disk
  store or re-issue ``get_or_create_collection`` on the hot query path.
* ``query_text_collection`` / ``query_image_collection`` optionally return the
  stored embeddings (``include_embeddings=True``) so callers can run MMR /
  reranking without re-embedding candidate chunks over the network.
"""

import threading
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Single unified collection for text + image vectors.
KNOWLEDGE_COLLECTION = "amref_knowledge"

# Discriminator values for the ``source_type`` metadata field.
SOURCE_TYPE_TEXT = "text"
SOURCE_TYPE_IMAGE = "image"

# Legacy collection names, kept so migration/inspection tooling can find and
# drain a pre-split store. Nothing on the query path reads these.
LEGACY_TEXT_COLLECTION = "amref_text_chunks"
LEGACY_IMAGE_COLLECTION = "amref_image_embeddings"

# Backwards-compatible aliases — older callers/scripts imported these names.
TEXT_COLLECTION = KNOWLEDGE_COLLECTION
IMAGE_COLLECTION = KNOWLEDGE_COLLECTION

# Retrieval fans text and image lookups into worker threads (see rag/retriever.py),
# so both the client and the collection handles can be requested concurrently.
# ``lru_cache`` does not serialise misses, which let two threads both enter
# ``PersistentClient()`` and corrupt Chroma's internal path registry
# (``KeyError: './data/chroma'``). A reentrant lock makes first-use atomic; after
# warmup every call hits the cached value and the lock is uncontended.
_init_lock = threading.RLock()
_client: chromadb.ClientAPI | None = None
_collections: dict[str, Collection] = {}


def _build_client() -> chromadb.ClientAPI:
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


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                _client = _build_client()
    return _client


def _get_collection(name: str) -> Collection:
    collection = _collections.get(name)
    if collection is None:
        with _init_lock:
            collection = _collections.get(name)
            if collection is None:
                collection = get_chroma_client().get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
                _collections[name] = collection
    return collection


def get_knowledge_collection() -> Collection:
    """The one collection holding both text and image vectors."""
    return _get_collection(KNOWLEDGE_COLLECTION)


# Backwards-compatible accessors — both now return the unified collection.
def get_text_collection() -> Collection:
    return get_knowledge_collection()


def get_image_collection() -> Collection:
    return get_knowledge_collection()


def _reset_collection_cache() -> None:
    """Drop cached collection handles (needed after delete/recreate)."""
    with _init_lock:
        _collections.clear()


def _with_source_type(metadatas: list[dict[str, Any]], source_type: str) -> list[dict[str, Any]]:
    """Stamp every metadata dict with its discriminator.

    Applied at the single write choke-point so no upsert path can forget it —
    an unstamped record would be invisible to every filtered query.
    """
    return [{**meta, "source_type": source_type} for meta in metadatas]


def upsert_text_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    collection = get_knowledge_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=_with_source_type(metadatas, SOURCE_TYPE_TEXT),
    )


def upsert_image_embeddings(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    collection = get_knowledge_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=_with_source_type(metadatas, SOURCE_TYPE_IMAGE),
    )


def _build_where(source_type: str, where: dict[str, Any] | None) -> dict[str, Any]:
    """Combine the source_type discriminator with an optional caller filter.

    Chroma requires an explicit ``$and`` when more than one field is
    constrained; a flat two-key dict is not a valid filter expression.
    """
    base = {"source_type": source_type}
    if not where:
        return base
    clauses = [base] + [{key: value} for key, value in where.items()]
    return {"$and": clauses}


def query_text_collection(
    query_embedding: list[float],
    n_results: int = 10,
    where: dict[str, Any] | None = None,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    collection = get_knowledge_collection()
    include = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include.append("embeddings")
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=_build_where(SOURCE_TYPE_TEXT, where),
        include=include,
    )


def query_image_collection(
    query_embedding: list[float],
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    collection = get_knowledge_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=_build_where(SOURCE_TYPE_IMAGE, where),
        include=["documents", "metadatas", "distances"],
    )


def clear_collections() -> None:
    """Drop the unified collection (and any legacy split collections)."""
    client = get_chroma_client()
    for name in (KNOWLEDGE_COLLECTION, LEGACY_TEXT_COLLECTION, LEGACY_IMAGE_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 — absent collection raises per-version types
            pass
    _reset_collection_cache()
    get_knowledge_collection()


def count_by_source_type() -> dict[str, int]:
    """Vector counts per source_type — used by health checks and ingest logs."""
    collection = get_knowledge_collection()
    counts: dict[str, int] = {}
    for source_type in (SOURCE_TYPE_TEXT, SOURCE_TYPE_IMAGE):
        try:
            result = collection.get(where={"source_type": source_type}, include=[])
            counts[source_type] = len(result.get("ids", []) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma count for source_type=%s failed: %s", source_type, exc)
            counts[source_type] = -1
    return counts


def check_embedding_dimension(expected_dim: int | None, log_only: bool = True) -> bool | None:
    """Inspect one stored embedding in the collection and compare its length.

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
    collection = get_knowledge_collection()

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
