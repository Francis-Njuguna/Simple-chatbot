"""ChromaDB client and collection management."""

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
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    logger.info("ChromaDB client initialized at %s", settings.chroma_persist_dir)
    return client


def get_text_collection() -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_image_collection() -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


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
) -> dict[str, Any]:
    collection = get_text_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
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
    get_text_collection()
    get_image_collection()
