"""Async database session management.

The engine is created once at import time from ``Settings.database_url``.
The resolved (redacted) connection string is logged here — at the single
point where ``create_async_engine`` is called — so every entrypoint that
touches the database (FastAPI app, scripts/ingest.py, scripts/inspect_kb.py)
shows exactly which credentials and host are in use, and gets the
split-brain warning from ``Settings.log_db_config()`` when a DATABASE_URL
override disagrees with the POSTGRES_* variables.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import get_settings
from backend.app.database.models import Base

_settings = get_settings()

# Surface the resolved DB target (redacted) before the first connection is
# attempted, so an InvalidPasswordError can immediately be correlated with
# the credentials/source that produced it.
_settings.log_db_config()

# Tune the async engine pool sizing using configured settings.  Using a
# small pool_size with a generous max_overflow lets the app handle spiky
# concurrency without creating too many long-lived connections.
engine = create_async_engine(
    _settings.database_url,
    echo=_settings.debug,
    pool_pre_ping=True,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_timeout=_settings.db_pool_timeout,
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
