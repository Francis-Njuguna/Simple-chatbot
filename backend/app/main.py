"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.app.api.routes import auth, chat, feedback, history, ingest
from backend.app.config import get_settings
from backend.app.database.session import init_db
from backend.app.utils.exceptions import AppError, app_error_to_http
from backend.app.utils.logging import setup_logging

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(settings.static_images_dir)
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(static_path)), name="images")

api_prefix = settings.api_prefix
app.include_router(auth.router, prefix=api_prefix)
app.include_router(chat.router, prefix=api_prefix)
app.include_router(feedback.router, prefix=api_prefix)
app.include_router(ingest.router, prefix=api_prefix)
app.include_router(history.router, prefix=api_prefix)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    http_exc = app_error_to_http(exc)
    return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})


@app.get("/health")
@limiter.limit(settings.rate_limit)
async def health(request: Request) -> dict:
    return {"status": "healthy", "app": settings.app_name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
