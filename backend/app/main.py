"""FastAPI application entry point.

Circular-import note
--------------------
The ``slowapi`` ``Limiter`` singleton used to be defined here, which caused an
``ImportError`` when route modules tried to import it:

    main.py  →  routes/chat.py  →  main.py   # partially-initialised module!

The limiter now lives in ``backend.app.core.limiter`` – a leaf module with no
back-references to this file – so both ``main.py`` and every route can import
it freely.

Startup warm-up
---------------
The single biggest source of the old "first request takes 30–60s" behaviour was
loading the SentenceTransformer model (and building the LLM HTTP client) lazily
*inside the first user request*. We now warm all heavy singletons during
``lifespan`` startup so the very first query is fast.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

# Shared limiter singleton – defined in a dedicated leaf module so that route
# handlers can import it without creating a circular dependency on main.py.
from backend.app.core.limiter import limiter  # noqa: E402  (must precede router imports)
from backend.app.api.routes import (
    auth,
    chat,
    feedback,
    history,
    ingest,
    observability,
)
from backend.app.config import get_settings
from backend.app.core.request_context import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    get_request_id,
    request_elapsed_ms,
)
from backend.app.database.session import init_db, pool_stats
from backend.app.utils.exceptions import (
    AppError,
    DatabaseUnavailableError,
    app_error_to_http,
)
from backend.app.utils.logging import get_logger, setup_logging
from backend.app.utils.metrics import get_metrics

settings = get_settings()
logger = get_logger(__name__)
metrics = get_metrics()


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------

async def _warmup() -> None:
    """Build and prime every heavy singleton so the first request is fast.

    * Loads the embedding backend (SentenceTransformer weights / Ollama pool)
      and runs one dummy embedding so torch / the model graph is fully warm.
    * Builds the LLM chat client (and its HTTP connection pool).
    * Opens the Chroma client + collection handles.

    Any failure here is logged but never blocks startup — a degraded service
    that lazily warms on first request is better than one that won't boot.
    """
    import anyio

    from backend.app.database.chroma import get_image_collection, get_text_collection
    from backend.app.rag.embeddings import get_embedding_service
    from backend.app.rag.llm import get_llm_service

    try:
        embedder = get_embedding_service()
        # Force the model to load + run a real forward pass off the event loop.
        await embedder.embed_query_async("warmup")
        logger.info("Warm-up: embedding model ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up: embedding warm-up failed (%s)", exc)

    try:
        get_llm_service()
        logger.info("Warm-up: LLM client ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up: LLM client build failed (%s)", exc)

    try:
        # Opening the collection handle does NOT load the HNSW index — the first
        # *query* does, and that measured ~7s on the text collection. Issue a
        # throwaway query against both so the first real request doesn't pay it.
        def _prime() -> None:
            from backend.app.database.chroma import (
                query_image_collection,
                query_text_collection,
            )

            zero = [0.0] * settings.embedding_dim
            query_text_collection(query_embedding=zero, n_results=1)
            query_image_collection(query_embedding=zero, n_results=1)

        await anyio.to_thread.run_sync(get_text_collection)
        await anyio.to_thread.run_sync(get_image_collection)
        await anyio.to_thread.run_sync(_prime)
        logger.info("Warm-up: Chroma collections ready (HNSW index primed).")

        # Best-effort embedding-dimension check: detect whether the stored
        # vectors in Chroma match the configured embedding model dimension.
        from backend.app.database.chroma import check_embedding_dimension

        try:
            dim_ok = await anyio.to_thread.run_sync(
                check_embedding_dimension, settings.embedding_dim, True
            )
            if dim_ok is False:
                logger.warning(
                    "Warm-up: embedding-dimension check flagged a mismatch — see previous CRITICAL log for details."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warm-up: embedding-dimension check failed (%s)", exc)

    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up: Chroma warm-up failed (%s)", exc)

    # Both of the following read the Chroma text corpus, so they must run after
    # the collection warm-up above.
    try:
        # BM25 needs a full scan of the text corpus to compute IDF. Small KB, but
        # doing it here keeps it off the first user query.
        from backend.app.rag.lexical import get_lexical_index

        await anyio.to_thread.run_sync(get_lexical_index().rebuild)
        logger.info("Warm-up: BM25 lexical index built.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up: lexical index build failed (%s)", exc)

    try:
        # The cross-encoder is by far the heaviest warm-up: a cold run that has
        # to download the weights measured ~46s. Paying it here rather than in
        # the first request is the difference between a 47s and a sub-second
        # first answer. Failure is non-fatal — the retriever falls back to
        # cosine ordering (see rag/reranker.py).
        from backend.app.rag.reranker import get_reranker

        reranker = get_reranker()
        await anyio.to_thread.run_sync(reranker.warmup)
        # A first predict() also compiles the graph; do one so query #1 doesn't.
        await anyio.to_thread.run_sync(reranker.score, "warmup", ["warmup passage"])
        logger.info("Warm-up: cross-encoder reranker ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up: cross-encoder warm-up failed (%s)", exc)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic.

    * Configures structured logging.
    * Logs the active (redacted) database URL so credential mismatches are
      immediately visible in the container logs.
    * Creates all SQL tables on first run (idempotent via ``CREATE IF NOT EXISTS``).
    * Warms heavy singletons (embedding model, LLM client, Chroma) so the first
      request does not pay cold-start costs.
    """
    setup_logging()

    # Surface the active DB credentials on every startup so operators can
    # immediately spot mismatches between .env, docker-compose, and the
    # running Postgres instance without having to dig through config files.
    settings.log_db_config()

    await init_db()
    await _warmup()
    yield

    # Graceful shutdown of the shared async HTTP client used for Ollama embeds.
    from backend.app.rag.embeddings import close_async_http_client

    await close_async_http_client()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach the limiter to app state so slowapi middleware can discover it.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this the browser hides X-Request-ID from JS, so a user reporting a
    # failed request has no id to quote.
    expose_headers=[REQUEST_ID_HEADER],
)

# Added last, so it runs FIRST: Starlette applies middleware in reverse
# registration order. The request id must be bound before anything else can log,
# and the X-Request-ID header must survive CORS.
app.add_middleware(RequestContextMiddleware)

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

static_path = Path(settings.static_images_dir)
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(static_path)), name="images")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

api_prefix = settings.api_prefix
app.include_router(auth.router, prefix=api_prefix)
app.include_router(chat.router, prefix=api_prefix)
app.include_router(feedback.router, prefix=api_prefix)
app.include_router(ingest.router, prefix=api_prefix)
app.include_router(history.router, prefix=api_prefix)
# Deliberately NOT under api_prefix: scrapers conventionally expect /metrics at
# the root, and Prometheus/OTel default scrape configs assume that path.
app.include_router(observability.router)

# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
#
# Why three handlers rather than one:
#
# * ``AppError``               — expected, already-classified failures. A 4xx is
#                                the caller's problem and is not worth a stack
#                                trace; a 5xx is ours and gets one.
# * ``SQLAlchemyTimeoutError`` — pool exhaustion that escaped ``db_scope``. Any
#                                code path still using the request-scoped
#                                ``get_db_session`` dependency raises this raw,
#                                and it previously matched no handler at all:
#                                Starlette turned it into a bare 500 with no log
#                                line, which is why a load test could produce 84
#                                failures and zero errors in ``app.log``.
# * ``Exception``              — the catch-all. Same reasoning: anything
#                                unhandled must leave a structured, greppable
#                                record behind rather than only a traceback on
#                                stdout that the log file never sees.
#
# Every handler logs the pool snapshot. Under saturation the proximate exception
# is often a symptom (a timeout somewhere else, a cancelled task) while the pool
# reading is the diagnosis, and it costs nothing to capture.


def _log_failure(request: Request, exc: BaseException, *, level: int = logging.ERROR) -> str:
    """Emit one structured record for a failed request. Returns its request id.

    Deliberately does not include the response body or any client payload — the
    id is the join key, and the request itself is already logged upstream.
    """
    state = request.scope.get("state") or {}
    request_id = state.get("request_id") or get_request_id()
    elapsed_ms = request_elapsed_ms(state)
    logger.log(
        level,
        "Request failed | id=%s | %s %s | exc=%s: %s | elapsed=%s | pool=%s",
        request_id,
        request.method,
        request.url.path,
        type(exc).__name__,
        exc,
        f"{elapsed_ms:.0f}ms" if elapsed_ms is not None else "unknown",
        pool_stats(),
        exc_info=exc,
    )
    return request_id


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    http_exc = app_error_to_http(exc)
    content: dict[str, object] = {"detail": http_exc.detail}
    if http_exc.status_code >= 500:
        # Server-side failure: log it with full context and hand the caller an
        # id they can quote. ``DatabaseUnavailableError`` arrives here as a 503.
        content["request_id"] = _log_failure(request, exc)
    return JSONResponse(status_code=http_exc.status_code, content=content)


@app.exception_handler(SQLAlchemyTimeoutError)
async def pool_timeout_handler(request: Request, exc: SQLAlchemyTimeoutError) -> JSONResponse:
    """Pool exhaustion that did not pass through ``db_scope``.

    Reported as 503, not 500: the request was well-formed and failed only
    because the server is saturated, so retrying is the correct client
    behaviour. The driver's message names pool sizes and can name the host, so
    it stays in the log and never reaches the response body.
    """
    metrics.counter(
        "db_pool_timeouts_total",
        labels={"scope": "unscoped"},
        help_text="Requests that failed waiting for a DB connection.",
    )
    request_id = _log_failure(request, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": DatabaseUnavailableError().message, "request_id": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _log_failure(request, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
@limiter.limit(settings.rate_limit)
async def health(request: Request) -> dict:  # noqa: ARG001
    """Lightweight liveness probe used by Docker / load-balancers."""
    return {"status": "healthy", "app": settings.app_name}


# ---------------------------------------------------------------------------
# Development entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
