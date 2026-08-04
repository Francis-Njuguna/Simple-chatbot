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

from pydantic import AliasChoices, Field, computed_field
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
    # LLM — OpenAI-compatible (primary), Gemini, Anthropic, or Ollama
    # Provider options: "openai" | "gemini" | "anthropic" | "ollama"
    # ------------------------------------------------------------------
    llm_provider: Literal["openai", "gemini", "anthropic", "ollama"] = Field(
        default="gemini", alias="LLM_PROVIDER"
    )

    # LLM runtime knobs (previously missing) — critical for stable builds.
    # These are referenced by backend.app.rag.llm and must exist to avoid
    # AttributeError during warm-up and per-request LLM client construction.
    # 2048 (not 1024): the synthesis prompt asks for a complete numbered
    # procedure plus caveats, which 1024 tokens truncates mid-answer.
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
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
    # Credential actually used by the Anthropic client. ANTHROPIC_AUTH_KEY is the
    # preferred name (it is what gateways/proxies issue); ANTHROPIC_API_KEY is
    # accepted as a fallback so existing deployments keep working.
    anthropic_auth_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_AUTH_KEY", "ANTHROPIC_API_KEY"),
    )
    # Optional base URL for an Anthropic-compatible gateway/proxy. When unset the
    # SDK default (api.anthropic.com) is used. Both names are accepted.
    anthropic_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_BASE_URL", "AUTH_BASE_URL"),
    )
    # Use the faster Haiku variant by default to reduce latency for short help-desk answers.
    anthropic_model: str = Field(default="claude-haiku-4-5", alias="ANTHROPIC_MODEL")

    # Google Gemini (optional)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    # Ollama (local fallback).
    # These must stay defined even when Ollama is unused: rag/llm.py and
    # rag/embeddings.py read settings.ollama_* unconditionally on their Ollama
    # branches, so commenting them out turned "Ollama not running" into an
    # AttributeError at provider-build time.
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

    # MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
    #
    # RETAINED FOR COMPATIBILITY, NO LONGER ON THE RETRIEVAL PATH. MMR used to
    # select the cross-encoder shortlist; it now selects nothing, because
    # optimising diversity *before* the only stage that judges relevance drops
    # chunks that answer the question. Measured on this KB, replacing it with
    # top-N by fused RRF rank moved recall@1 18/20 → 19/20 and recall@k 19/20 →
    # 20/20 with off-topic leakage unchanged (see the Stage 4 comment in
    # retriever.py). Redundancy is handled downstream instead, by exact-text
    # dedup and by group_adjacent_chunks merging sibling chunks into one block.
    #
    # _mmr_select_vectorised is kept as a tested utility for callers that want a
    # diversity-aware selection over an already-relevant set.
    mmr_diversity: float = Field(default=0.7, alias="MMR_DIVERSITY")

    # ------------------------------------------------------------------
    # Hybrid retrieval (BM25 + vector)
    #
    # Vector search alone misses exact-token queries ("SMOWL", "VAS", an error
    # code); BM25 alone misses paraphrases. Results are fused with Reciprocal
    # Rank Fusion, which combines *rankings* rather than raw scores, so the two
    # very differently-scaled systems need no score normalisation.
    # ------------------------------------------------------------------
    hybrid_search_enabled: bool = Field(default=True, alias="HYBRID_SEARCH_ENABLED")
    # Relative pull of each ranking in the fusion. Vector is weighted higher
    # because paraphrased questions are the common case here.
    hybrid_vector_weight: float = Field(default=0.6, alias="HYBRID_VECTOR_WEIGHT")
    hybrid_bm25_weight: float = Field(default=0.4, alias="HYBRID_BM25_WEIGHT")
    # RRF damping constant. 60 is the value from the original RRF paper and
    # keeps any single engine's #1 hit from dominating the fused order.
    rrf_k: int = Field(default=60, alias="RRF_K")

    # Candidates pulled from each engine before the shortlist narrows them down.
    # Must exceed rerank_shortlist, or the shortlist is the whole pool and the
    # wider retrieval buys nothing.
    #
    # 40 (was 30): multi-query retrieval unions candidates from several query
    # variants, and the pool has to be wide enough that a chunk only the third
    # variant found still gets in. Cost is bounded — widening the pool adds
    # Chroma read time, not cross-encoder time, which is what actually dominates
    # latency (the shortlist gates that).
    retrieval_candidate_pool: int = Field(default=40, alias="RETRIEVAL_CANDIDATE_POOL")
    # How many fused candidates go to the (more expensive) cross-encoder.
    # 16 (was 12): a wider pool is pointless if the shortlist re-narrows it
    # before the cross-encoder — which is the only stage that can tell a
    # genuine answer from a vocabulary match — but this is the stage that costs
    # real milliseconds, so it grows more conservatively than the pool.
    #
    # Named MMR_SHORTLIST historically, when MMR chose these. It is now a plain
    # top-N cut of the fused RRF ranking; the env alias is kept so existing
    # deployments do not silently fall back to the default.
    rerank_shortlist: int = Field(
        default=16, validation_alias=AliasChoices("RERANK_SHORTLIST", "MMR_SHORTLIST")
    )

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    #
    # The bi-encoder scores query and chunk independently; a cross-encoder reads
    # both together and is markedly better at ordering. It only runs on the MMR
    # shortlist, so the cost is bounded. Disabled automatically if the model
    # cannot be loaded — retrieval falls back to cosine order rather than fail.
    # ------------------------------------------------------------------
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )
    # Absolute relevance gate on the cross-encoder logit. Unlike cosine, this
    # score reflects whether a passage *answers* the query, so it can reject
    # material that merely shares vocabulary with it.
    #
    # Calibrated against this KB with ms-marco-MiniLM-L-6-v2: chunks that
    # genuinely answered a question scored about -5..+4, while a question the
    # articles do not cover ("how do I submit an assignment in Moodle?") scored
    # ≈ -11 on every chunk despite clearing the cosine floor. -8.0 sits in the
    # empty band between those two clusters. Note this is model-specific — a
    # different rerank_model produces a different scale and needs re-calibrating.
    # Set to a large negative number (e.g. -1e9) to disable the gate.
    rerank_min_score: float = Field(default=-8.0, alias="RERANK_MIN_SCORE")

    # Share of the final ORDERING given to the cross-encoder, the remainder going
    # to the fused RRF rank. The gate above is unaffected — it is always the
    # cross-encoder alone.
    #
    # Splitting these apart is the point: this model is excellent at judging
    # whether a passage answers the question at all (it declines every genuinely
    # off-topic query) and noisy at ranking two passages that both do. Measured
    # on "Can't access LMS", ordering by its score alone put the Microsoft Teams
    # chunk (+1.27) above the actual "How to login to LMS" chunk (-1.43) and
    # pushed the right answer past top_k, while RRF had it at #1. Over 30
    # paraphrase/synonym/typo queries, any blend recovered recall@k 29/30 → 30/30
    # with off-topic precision unchanged; 0.5 is the midpoint rather than a value
    # fitted to that set. 1.0 restores pure cross-encoder ordering.
    rerank_order_weight: float = Field(default=0.5, alias="RERANK_ORDER_WEIGHT")

    # How many phrasings of the question the cross-encoder scores, keeping the
    # best score per chunk. This is a recall/precision dial, not a free win.
    #
    # This cross-encoder is very sensitive to surface wording: on this KB the
    # same six login passages score +5.5/+7.6 for "LMS login" but -8.7/-10.8 for
    # "Moodle login", because the article is titled "How to login to LMS". With
    # one phrasing the gate is partly a test of whether the user guessed the
    # article's own vocabulary.
    #
    # Measured over 21 on-topic and 8 off-topic queries:
    #   1 form  → 19/21 on-topic (90.5%), 8/8 off-topic blocked
    #   2 forms → 21/21 on-topic (100%),  8/8 off-topic blocked
    #   3 forms → 21/21 on-topic,         6/8 off-topic ("weather tomorrow"
    #             and "quantum entanglement" cleared the gate at -7.0/-7.1)
    # Each extra phrasing is another attempt against a fixed threshold, so
    # raising this past 2 buys nothing and costs off-topic precision.
    rerank_query_forms: int = Field(default=2, alias="RERANK_QUERY_FORMS")

    # ------------------------------------------------------------------
    # Query preprocessing + retrieval cache
    # ------------------------------------------------------------------
    # Fuzzy-corrects domain terms ("smwol" → "smowl") against the KB vocabulary.
    # Purely local (difflib) — no LLM call, so it costs well under a millisecond.
    query_rewrite_enabled: bool = Field(default=True, alias="QUERY_REWRITE_ENABLED")

    # --- Query preprocessing stages (see rag/query_processing.py) ---
    # Each stage is separately switchable so a recall change can be attributed
    # to one specific stage in the benchmark rather than to "preprocessing".
    #
    # Normalisation: lowercase, strip punctuation, correct spelling, expand
    # abbreviations ("pwd" → "password"), canonicalise acronym casing.
    query_normalization_enabled: bool = Field(
        default=True, alias="QUERY_NORMALIZATION_ENABLED"
    )
    # Synonym expansion feeds BM25 ONLY. The vector query keeps the normalised
    # text and the cross-encoder keeps the user's original wording — appending
    # a bag of synonyms to either degrades it (see query_processing docstring).
    query_synonym_expansion_enabled: bool = Field(
        default=True, alias="QUERY_SYNONYM_EXPANSION_ENABLED"
    )
    # Multi-query: embed several paraphrases and fuse their rankings, so
    # retrieval stops depending on the user's exact phrasing.
    multi_query_enabled: bool = Field(default=True, alias="MULTI_QUERY_ENABLED")
    # Paraphrases *in addition to* the normalised query. Each costs one
    # embedding (~5-15ms local) plus one Chroma read; 4 keeps the whole
    # preprocessing budget well inside the 3s latency target.
    multi_query_variants: int = Field(default=4, alias="MULTI_QUERY_VARIANTS")
    # Weight of a variant's ranking relative to the primary query's in RRF.
    # Below 1.0 because a paraphrase is derived evidence: it should be able to
    # rescue a chunk the original phrasing missed, but never outvote it.
    multi_query_variant_weight: float = Field(
        default=0.5, alias="MULTI_QUERY_VARIANT_WEIGHT"
    )
    # BM25 fuzzy token matching: map an out-of-vocabulary query token to its
    # nearest *corpus* term before scoring. Catches typos the explicit
    # correction map never enumerated.
    lexical_fuzzy_enabled: bool = Field(default=True, alias="LEXICAL_FUZZY_ENABLED")

    # Merge chunks from the same article into one context block, in document
    # order, before the prompt is built. Adjacent chunks split mid-procedure
    # otherwise arrive as disconnected fragments in arbitrary rerank order.
    group_adjacent_chunks: bool = Field(default=True, alias="GROUP_ADJACENT_CHUNKS")

    # Seconds to cache a full retrieval result keyed by normalised query.
    retrieval_cache_ttl: int = Field(default=300, alias="RETRIEVAL_CACHE_TTL")
    retrieval_cache_size: int = Field(default=256, alias="RETRIEVAL_CACHE_SIZE")

    # Emit per-query retrieval diagnostics (candidates, scores, fusion, final
    # context). Verbose and can echo user queries into logs — keep OFF in
    # production. Forced off when APP_ENV is production-like; see
    # ``retrieval_debug_active``.
    retrieval_debug: bool = Field(default=False, alias="RETRIEVAL_DEBUG")

    # Floor a chunk's score must clear to be handed to the LLM. Below this the
    # context is treated as a miss so the model declines instead of answering
    # from loosely-related material.
    min_relevance_score: float = Field(default=0.25, alias="MIN_RELEVANCE_SCORE")
    min_image_score: float = Field(default=0.30, alias="MIN_IMAGE_SCORE")

    # Seconds to cache article/image metadata hydrated from PostgreSQL. Keeps
    # the chat hot path off the database; a re-ingest in this process clears the
    # cache immediately, so this only bounds staleness for *other* processes.
    metadata_cache_ttl: int = Field(default=300, alias="METADATA_CACHE_TTL")

    # Extractive summary shape, computed during ingestion (no LLM call).
    summary_max_sentences: int = Field(default=5, alias="SUMMARY_MAX_SENTENCES")
    summary_max_chars: int = Field(default=1200, alias="SUMMARY_MAX_CHARS")

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

    # How many articles to fetch at once during a crawl. Bounded so ingestion
    # does not hammer the KB host; raise only if the host tolerates it.
    crawl_concurrency: int = Field(default=6, alias="CRAWL_CONCURRENCY")

    # Guard against a partial crawl wiping a good knowledge base. A force
    # re-ingest clears the collection before writing, so if the crawler returns
    # far fewer articles than Postgres already knows about — one TLS blip is
    # enough, since crawl_all() swallows per-article errors — the clear would
    # discard content the crawl simply failed to re-fetch. Below this fraction
    # of the known article count, ingestion refuses and leaves the KB intact.
    # Set to 0 to disable (e.g. when articles were genuinely removed upstream).
    reingest_min_coverage: float = Field(default=0.5, alias="REINGEST_MIN_COVERAGE")

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
    def is_production(self) -> bool:
        return (self.app_env or "").strip().lower() in {"production", "prod", "staging"}

    @property
    def retrieval_debug_active(self) -> bool:
        """Whether to emit per-query retrieval diagnostics.

        RETRIEVAL_DEBUG is honoured only outside production-like environments.
        The dumps include the raw user query and full chunk bodies, which is
        exactly what you want when diagnosing a bad answer and exactly what you
        do not want written to a production log.
        """
        return bool(self.retrieval_debug) and not self.is_production

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

    # Run `alembic upgrade head` automatically on application startup. Handy on
    # Railway/Docker where there is no separate migration step. Set to false when
    # migrations are applied by a dedicated deploy job.
    db_auto_migrate: bool = Field(default=True, alias="DB_AUTO_MIGRATE")

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
