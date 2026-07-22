"""Application configuration via environment variables.

Credential precedence for the PostgreSQL connection
----------------------------------------------------
The connection URL is resolved in this order (first match wins):

1. ``DATABASE_URL`` env var — if set, it is used verbatim and the four
   ``POSTGRES_*`` variables are only used for display / healthcheck purposes.
   NOTE: pydantic-settings gives *process environment variables* priority
   over the ``.env`` file, so a stale ``DATABASE_URL`` exported in your
   shell silently overrides everything in ``.env``.
2. Assembled from ``POSTGRES_USER``, ``POSTGRES_PASSWORD``,
   ``POSTGRES_HOST``, ``POSTGRES_PORT``, and ``POSTGRES_DB``.

Local development keeps ONLY the four POSTGRES_* parts in ``.env`` (single
source of truth).  ``DATABASE_URL`` is reserved for environments that inject
it explicitly (docker-compose, Railway).  ``log_db_config()`` prints which
source was used and warns loudly when the two sources disagree.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Amref Help Desk RAG", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    streamlit_port: int = Field(default=8501, alias="STREAMLIT_PORT")

    secret_key: str = Field(default="change-me", alias="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=1440, alias="JWT_EXPIRE_MINUTES")
    rate_limit: str = Field(default="30/minute", alias="RATE_LIMIT")

    # ------------------------------------------------------------------
    # PostgreSQL — individual credential components
    # These must match the POSTGRES_* env vars given to the postgres
    # Docker service in docker-compose.yml.
    # ------------------------------------------------------------------
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="amref", alias="POSTGRES_USER")
    postgres_password: str = Field(default="amref_secret", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="amref_helpdesk", alias="POSTGRES_DB")

    # Optional pre-assembled URL override.  When set it takes precedence over
    # the four POSTGRES_* parts so that tooling that only speaks DATABASE_URL
    # (e.g. Alembic, Railway's injected variable) and the app share one URL.
    # Deliberately NOT set in .env for local runs — see module docstring.
    database_url_override: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # ------------------------------------------------------------------
    # ChromaDB
    #
    # CHROMA_MODE selects how the vector store is reached:
    #   "auto"       (default) — use HttpClient if CHROMA_SERVER_HOST is set,
    #                            otherwise fall back to a local PersistentClient
    #                            reading CHROMA_PERSIST_DIR (baked image / volume).
    #   "persistent" — always use the on-disk PersistentClient.
    #   "http"       — always connect to a standalone Chroma server over HTTP.
    #
    # CHROMA_SERVER_HOST / CHROMA_SERVER_PORT point at a standalone Chroma
    # service (e.g. a separate Railway service). When unset, persistent mode is
    # used. (CHROMA_HOST/CHROMA_PORT are kept for backwards compatibility.)
    # ------------------------------------------------------------------
    chroma_mode: Literal["auto", "persistent", "http"] = Field(
        default="auto", alias="CHROMA_MODE"
    )
    chroma_server_host: Optional[str] = Field(default=None, alias="CHROMA_SERVER_HOST")
    chroma_server_port: int = Field(default=8000, alias="CHROMA_SERVER_PORT")
    chroma_server_ssl: bool = Field(default=False, alias="CHROMA_SERVER_SSL")
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8001, alias="CHROMA_PORT")
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")

    # ------------------------------------------------------------------
    # LLM — OpenAI-compatible (primary)
    # Provider options: "openai" | "anthropic" | "ollama"
    # ------------------------------------------------------------------
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        default="ollama", alias="LLM_PROVIDER"
    )

    # LLM runtime knobs (previously missing) — critical for stable builds.
    # These are referenced by backend.app.rag.llm and must exist to avoid
    # AttributeError during warm-up and per-request LLM client construction.
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_timeout: int = Field(default=30, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")

    # OpenAI-compatible provider (default)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    # Optional API base for OpenAI-compatible endpoints (e.g. Qwen providers,
    # local Ollama OpenAI-compatible servers). When set, the runtime will target
    # this base URL instead of api.openai.com. If the local endpoint is unauthenticated,
    # OPENAI_API_KEY may be omitted and a dummy key is used for compatibility.
    openai_api_base: str | None = Field(default=None, alias="OPENAI_API_BASE")

    # Anthropic (optional fallback)
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Use the faster Haiku variant by default to reduce latency for short help-desk answers.
    anthropic_model: str = Field(default="claude-haiku-4-5", alias="ANTHROPIC_MODEL")

    # Ollama (local fallback)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:4b", alias="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=120, alias="OLLAMA_TIMEOUT")
    ollama_keep_alive: str | int | None = Field(default="-1", alias="OLLAMA_KEEP_ALIVE")
 
    # ------------------------------------------------------------------
    # Embeddings
    # Provider options: "ollama" | "sentence-transformers"
    # ------------------------------------------------------------------
    # Default to the sentence-transformers provider to match EMBEDDING_MODEL and
    # to avoid an accidental mismatch between provider vs model used at build time.
    embedding_provider: Literal["ollama", "sentence-transformers"] = Field(
        default="sentence-transformers", alias="EMBEDDING_PROVIDER"
    )
    # When using sentence-transformers this should be a HF model id. When using
    # Ollama, the Ollama model name in OLLAMA_EMBEDDING_MODEL is used instead.
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL"
    )

    # Computed helper: expected embedding dimensionality for the configured
    # provider+model. Useful for runtime checks and operator logs.
    @property
    def embedding_dim(self) -> int | None:
        """Return the expected embedding vector dimension for the active provider.

        Returns None if the dimension is unknown for the configured model.
        """
        if self.embedding_provider == "ollama":
            if self.ollama_embedding_model == "nomic-embed-text":
                return 768
            return None
        if self.embedding_provider == "sentence-transformers":
            if "all-MiniLM-L6-v2" in (self.embedding_model or ""):
                return 384
            return None
        return None

    # ------------------------------------------------------------------
    # RAG pipeline
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(default=5, alias="TOP_K_RETRIEVAL")
    top_k_images: int = Field(default=3, alias="TOP_K_IMAGES")
    mmr_diversity: float = Field(default=0.3, alias="MMR_DIVERSITY")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")

    kb_base_url: str = Field(default="https://helpdesk.amref.ac.ke", alias="KB_BASE_URL")
    kb_index_url: str = Field(
        default="https://helpdesk.amref.ac.ke/knowledgebase.php", alias="KB_INDEX_URL"
    )

    # ------------------------------------------------------------------
    # Crawler TLS configuration
    #
    # The Amref help desk host frequently serves an INCOMPLETE certificate
    # chain (it omits the intermediate CA cert), which yields:
    #   [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate
    # No CA bundle (certifi or system) can verify such a chain because the
    # intermediate needed to link the leaf to a trusted root is missing.
    #
    # KB_CA_BUNDLE — path to a custom PEM bundle that contains the missing
    #   intermediate CA (recommended fix; keeps verification ON).
    # KB_VERIFY_SSL — set to false as a last-resort escape hatch to skip
    #   verification for the crawler only (does NOT affect the DB or LLM
    #   clients).  Defaults to true.
    # ------------------------------------------------------------------
    kb_ca_bundle: Optional[str] = Field(default=None, alias="KB_CA_BUNDLE")
    kb_verify_ssl: bool = Field(default=True, alias="KB_VERIFY_SSL")

    # Explicit list of category IDs to crawl.
    # Comma-separated string in env, e.g. KB_CATEGORY_IDS=1,2,3,5
    # When set, the crawler uses ONLY these categories instead of
    # discovering them dynamically from the index page.
    kb_category_ids: str = Field(
        default="1,2,3,5,6,7,8,9,10,11,12,13,14,15",
        alias="KB_CATEGORY_IDS",
    )

    data_dir: str = Field(default="./data", alias="DATA_DIR")
    raw_data_dir: str = Field(default="./data/raw", alias="RAW_DATA_DIR")
    processed_data_dir: str = Field(default="./data/processed", alias="PROCESSED_DATA_DIR")
    images_dir: str = Field(default="./data/images", alias="IMAGES_DIR")
    static_images_dir: str = Field(
        default="./backend/app/static/images", alias="STATIC_IMAGES_DIR"
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", alias="LOG_FILE")

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    @property
    def kb_category_id_list(self) -> list[str]:
        """Return KB_CATEGORY_IDS as a clean list of string IDs."""
        return [c.strip() for c in self.kb_category_ids.split(",") if c.strip()]

    @property
    def use_chroma_http(self) -> bool:
        """Whether to connect to a standalone Chroma server over HTTP.

        - CHROMA_MODE=http       → always HTTP (requires CHROMA_SERVER_HOST).
        - CHROMA_MODE=persistent → never HTTP.
        - CHROMA_MODE=auto       → HTTP only when CHROMA_SERVER_HOST is set.
        """
        if self.chroma_mode == "http":
            return True
        if self.chroma_mode == "persistent":
            return False
        # auto
        return bool(self.chroma_server_host)

    # ------------------------------------------------------------------
    # Computed connection URLs
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[misc]
    @property
    def database_url(self) -> str:
        """Async SQLAlchemy URL (postgresql+asyncpg://).

        Returns ``DATABASE_URL`` verbatim when it is set in the environment,
        falling back to assembling the URL from the four POSTGRES_* parts.
        The asyncpg driver prefix is enforced so SQLAlchemy always gets the
        correct dialect regardless of how the env var was written.
        """
        if self.database_url_override:
            url = self.database_url_override
            # Normalise bare "postgresql://" → "postgresql+asyncpg://"
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def sync_database_url(self) -> str:
        """Synchronous psycopg2 URL for Alembic / sync tooling."""
        if self.database_url_override:
            url = self.database_url_override
            # Strip asyncpg driver if present so sync tools don't choke.
            url = url.replace("postgresql+asyncpg://", "postgresql://")
            return url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_source(self) -> str:
        """Human-readable description of where database_url came from."""
        if self.database_url_override:
            return "DATABASE_URL (override — env var or .env)"
        return "POSTGRES_* parts (assembled)"

    def redacted_database_url(self) -> str:
        """Return database_url with the *actual* password in the URL masked.

        Unlike a naive ``replace(postgres_password, ...)`` this parses the
        URL, so an override URL carrying a DIFFERENT password than
        POSTGRES_PASSWORD is still redacted and never leaks into logs.
        """
        url = self.database_url
        try:
            parts = urlsplit(url)
            if parts.password:
                masked_netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
                url = url.replace(parts.netloc, masked_netloc, 1)
        except ValueError:
            # Unparseable URL — redact everything between "//" and "@" defensively.
            head, sep, tail = url.partition("@")
            if sep:
                scheme_end = head.find("//") + 2
                url = head[:scheme_end] + "***:***" + sep + tail
        return url

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.images_dir,
            self.static_images_dir,
            self.chroma_persist_dir,
            str(Path(self.log_file).parent),
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Database connection pool tuning (important for remote DBs like Railway)
    # ------------------------------------------------------------------
    # Adjust these for production to allow higher concurrency without serialising
    # connections when the DB is in another region.
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")

    def log_db_config(self) -> None:
        """Emit a redacted summary of the active DB config to stdout.

        Called at engine creation (backend/app/database/session.py) so EVERY
        entrypoint — the FastAPI app, scripts/ingest.py, scripts/inspect_kb.py
        — surfaces which credentials are actually in use, without exposing
        the password.  Also detects "split-brain" configuration: when a
        DATABASE_URL override disagrees with the POSTGRES_* parts.
        """
        print(
            f"[config] DB config → host={self.postgres_host} "
            f"port={self.postgres_port} user={self.postgres_user} "
            f"db={self.postgres_db} password={'set' if self.postgres_password else 'EMPTY'}"
        )
        print(f"[config] database_url source → {self.database_url_source}")
        print(f"[config] database_url (redacted) → {self.redacted_database_url()}")

        if not self.database_url_override:
            return

        # Split-brain detection: compare the override URL against the
        # POSTGRES_* parts and warn on every mismatched component.
        try:
            parts = urlsplit(self.database_url)
        except ValueError:
            print("[config] WARNING: DATABASE_URL could not be parsed for validation.")
            return

        mismatches: list[str] = []
        if parts.username and parts.username != self.postgres_user:
            mismatches.append(f"user ('{parts.username}' != '{self.postgres_user}')")
        if parts.password and parts.password != self.postgres_password:
            mismatches.append("password (values differ — redacted)")
        if parts.hostname and parts.hostname != self.postgres_host:
            mismatches.append(f"host ('{parts.hostname}' != '{self.postgres_host}')")
        if parts.port and parts.port != self.postgres_port:
            mismatches.append(f"port ({parts.port} != {self.postgres_port})")
        db_name = (parts.path or "").lstrip("/")
        if db_name and db_name != self.postgres_db:
            mismatches.append(f"database ('{db_name}' != '{self.postgres_db}')")

        if mismatches:
            print(
                "[config] WARNING: split-brain DB configuration detected!\n"
                "[config]   DATABASE_URL overrides the POSTGRES_* variables but "
                "disagrees with them on: " + ", ".join(mismatches) + "\n"
                "[config]   If this is a local run, a stale DATABASE_URL is likely "
                "exported in your shell or left in .env.\n"
                "[config]   Fix: `unset DATABASE_URL` (or remove it from .env) so the "
                "POSTGRES_* parts are used, or update it to match."
            )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
