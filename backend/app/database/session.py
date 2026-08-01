"""Async database session management.

The engine is created once at import time from ``Settings.database_url``.
The resolved (redacted) connection string is logged here — at the single
point where ``create_async_engine`` is called — so every entrypoint that
touches the database (FastAPI app, scripts/ingest.py, scripts/inspect_kb.py)
shows exactly which credentials and host are in use, and gets the
split-brain warning from ``Settings.log_db_config()`` when a DATABASE_URL
override disagrees with the POSTGRES_* variables.

Schema management
-----------------
Schema changes go through **Alembic**, never ``create_all()``. ``run_migrations``
brings the database to ``head``; for a database that predates Alembic (tables
already created by the old ``create_all`` call) it first stamps the baseline
revision so no table is ever recreated or dropped.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import anyio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

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

# Revision that reproduces the pre-Alembic ``create_all()`` schema. A database
# that already has the tables but no ``alembic_version`` row is stamped here.
BASELINE_REVISION = "0001a0baseline"

# Arbitrary but fixed key for the Postgres advisory lock that serialises
# migrations. Multiple app workers (or an app + an ingest script) can start at
# the same instant; without this they race to run the same DDL.
_MIGRATION_LOCK_KEY = 0x414D5245  # "AMRE"


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _alembic_config():
    """Build an Alembic config pointed at this project's ``alembic/`` dir."""
    from alembic.config import Config

    root = Path(__file__).resolve().parents[3]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", _settings.sync_database_url)
    return config


async def _needs_baseline_stamp() -> bool:
    """True when tables exist but Alembic has never tracked this database."""
    async with engine.connect() as conn:
        has_version_table = await conn.scalar(
            text("SELECT to_regclass('public.alembic_version') IS NOT NULL")
        )
        if has_version_table:
            return False
        # No alembic_version. If the app tables already exist, this database was
        # built by the old create_all() path and must be adopted, not rebuilt.
        return bool(
            await conn.scalar(text("SELECT to_regclass('public.document_metadata') IS NOT NULL"))
        )


def _run_upgrade_sync(stamp_baseline: bool) -> None:
    """Blocking Alembic call — always invoked via ``anyio.to_thread``."""
    from alembic import command

    config = _alembic_config()
    if stamp_baseline:
        logger.info(
            "Existing pre-Alembic database detected — stamping baseline revision %s",
            BASELINE_REVISION,
        )
        command.stamp(config, BASELINE_REVISION)
    command.upgrade(config, "head")


async def run_migrations() -> None:
    """Upgrade the database to ``head``.

    Serialised across processes with a Postgres advisory lock so concurrent
    workers cannot run the same DDL simultaneously. Alembic itself is sync, so
    the work happens in a worker thread and never blocks the event loop.
    """
    if not _settings.db_auto_migrate:
        logger.info("DB_AUTO_MIGRATE is false — skipping automatic migrations.")
        return

    async with engine.connect() as conn:
        await conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": _MIGRATION_LOCK_KEY})
        await conn.commit()
        try:
            stamp_baseline = await _needs_baseline_stamp()
            await anyio.to_thread.run_sync(_run_upgrade_sync, stamp_baseline)
            logger.info("Database migrations are up to date (head).")
        finally:
            await conn.execute(
                text("SELECT pg_advisory_unlock(:key)"), {"key": _MIGRATION_LOCK_KEY}
            )
            await conn.commit()


async def init_db() -> None:
    """Backwards-compatible alias for :func:`run_migrations`.

    Kept so existing entrypoints (scripts/ingest.py, older deploy scripts)
    keep working. It no longer calls ``create_all`` — Alembic owns the schema.
    """
    await run_migrations()
