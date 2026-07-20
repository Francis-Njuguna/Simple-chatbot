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

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Shared limiter singleton – defined in a dedicated leaf module so that route
# handlers can import it without creating a circular dependency on main.py.
from backend.app.core.limiter import limiter  # noqa: E402  (must precede router imports)
from backend.app.api.routes import auth, chat, feedback, history, ingest
from backend.app.config import get_settings
from backend.app.database.session import init_db
from backend.app.utils.exceptions import AppError, app_error_to_http
from backend.app.utils.logging import get_logger, setup_logging

settings = get_settings()
logger = get_logger(__name__)


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

    from backend.app.database.chroma import get_text_collection
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
        await anyio.to_thread.run_sync(get_text_collection)
        logger.info("Warm-up: Chroma collection ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up: Chroma warm-up failed (%s)", exc)


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
)

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

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    http_exc = app_error_to_http(exc)
    return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})


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
