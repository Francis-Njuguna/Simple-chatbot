"""FastAPI application entry point.

Circular-import note
--------------------
The ``slowapi`` ``Limiter`` singleton used to be defined here, which caused an
``ImportError`` when route modules tried to import it:

    main.py  →  routes/chat.py  →  main.py   # partially-initialised module!

The limiter now lives in ``backend.app.core.limiter`` – a leaf module with no
back-references to this file – so both ``main.py`` and every route can import
it freely.
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
from backend.app.utils.logging import setup_logging

settings = get_settings()


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
    """
    setup_logging()

    # Surface the active DB credentials on every startup so operators can
    # immediately spot mismatches between .env, docker-compose, and the
    # running Postgres instance without having to dig through config files.
    settings.log_db_config()

    await init_db()
    yield


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
