"""CLI script to run knowledge base ingestion."""

import asyncio
import os
import ssl
import sys
from pathlib import Path

# Allow running from the project root without the package being installed.
# Inside Docker this is redundant (pip install -e . handles it) but keeps
# the script working for local runs too — same pattern as verify_db_credentials.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import certifi

# ---------------------------------------------------------------------------
# Force certifi's CA bundle for the WHOLE process, before any HTTP client is
# imported/instantiated. Setting these env vars makes httpx, requests, aiohttp
# and stdlib ssl all default to certifi rather than the (possibly empty) system
# trust store — which matters inside slim Docker images.
# ---------------------------------------------------------------------------
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from backend.app.database.session import async_session_factory, init_db
from backend.app.ingest import crawler as crawler_module
from backend.app.ingest.pipeline import IngestionPipeline
from backend.app.rag.embeddings import EmbeddingService
from backend.app.utils.logging import setup_logging


def _print_tls_diagnostics() -> None:
    """Prove which crawler file and CA material this process actually uses."""
    print("=== ingest.py TLS DIAGNOSTICS ===")
    print(f"crawler module loaded from : {crawler_module.__file__}")
    print(f"certifi.where()            : {certifi.where()}")
    print(f"SSL_CERT_FILE env          : {os.environ.get('SSL_CERT_FILE')}")
    print(f"REQUESTS_CA_BUNDLE env     : {os.environ.get('REQUESTS_CA_BUNDLE')}")
    print(f"ssl default verify paths   : {ssl.get_default_verify_paths()}")
    print("=================================")


async def main() -> None:
    setup_logging()
    _print_tls_diagnostics()
    await init_db()
    async with async_session_factory() as session:
        pipeline = IngestionPipeline(session, EmbeddingService())
        result = await pipeline.run(force=False, include_images=True)
        await session.commit()
        print(result)



if __name__ == "__main__":
    asyncio.run(main())
