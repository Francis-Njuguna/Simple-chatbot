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
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as SQLTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import get_settings
from backend.app.utils.exceptions import DatabaseUnavailableError
from backend.app.utils.logging import get_logger
from backend.app.utils.metrics import get_metrics

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


# ---------------------------------------------------------------------------
# Short-lived scoped sessions
# ---------------------------------------------------------------------------
#
# ``get_db_session`` above is a FastAPI ``yield`` dependency, so the session it
# produces lives for the WHOLE request — FastAPI only unwinds the generator
# after the response has been sent. Any handler that does DB work and then does
# something slow (embedding, reranking, calling an external LLM) therefore pins
# a pooled connection for its entire duration. With pool_size(5) +
# max_overflow(10) that caps the server at 15 in-flight requests regardless of
# how much CPU is idle, and request 16 blocks for pool_timeout then fails.
#
# ``db_scope`` is the alternative for those paths: acquire, do only the DB work,
# commit, release. It is deliberately NOT a FastAPI dependency — the point is
# that its lifetime is bounded by the ``async with`` block, not by the request.


def pool_stats() -> dict[str, int | float]:
    """Snapshot of the connection pool. Safe to log and expose.

    Contains sizes and counts only — never a DSN, host or credential.
    ``NullPool`` (used in some test setups) implements none of these, so every
    read is guarded rather than assumed.
    """
    pool = engine.pool
    stats: dict[str, int | float] = {
        "pool_size": _settings.db_pool_size,
        "max_overflow": _settings.db_max_overflow,
        "pool_timeout_s": _settings.db_pool_timeout,
        "ceiling": _settings.db_pool_size + _settings.db_max_overflow,
    }
    for label, attr in (
        ("checked_out", "checkedout"),
        ("checked_in", "checkedin"),
        ("overflow", "overflow"),
    ):
        getter = getattr(pool, attr, None)
        if callable(getter):
            try:
                stats[label] = getter()
            except Exception:  # noqa: BLE001 — diagnostics must never raise
                pass
    if "checked_out" in stats:
        stats["available"] = max(0, int(stats["ceiling"]) - int(stats["checked_out"]))
    return stats


def publish_pool_gauges() -> None:
    """Copy the current pool snapshot into the metrics registry as gauges.

    Called at scrape time rather than on a timer: ``checkedout`` is an instant
    reading, and a value sampled by a background task seconds ago would be worse
    than no value at all when the question is "was the pool saturated *now*".

    Naming: keys already prefixed with ``pool_`` are not prefixed twice, so the
    exported series are ``db_pool_size``, ``db_pool_checked_out``, … rather than
    ``db_pool_pool_size``.
    """
    metrics = get_metrics()
    for key, value in pool_stats().items():
        name = key[len("pool_"):] if key.startswith("pool_") else key
        metrics.gauge(
            f"db_pool_{name}",
            float(value),
            help_text="PostgreSQL connection pool: " + name.replace("_", " "),
        )


@asynccontextmanager
async def db_scope(label: str = "db") -> AsyncGenerator[AsyncSession, None]:
    """Acquire a connection, run the block, commit, release. Nothing else.

    Wrap ONLY database work in this. Never wrap an LLM call, an embedding, a
    rerank or response serialisation — holding the connection across those is
    precisely the exhaustion this exists to prevent.

    Emits two measurements per use:
      * ``db_acquire_ms`` — time spent waiting for a pooled connection. This is
        the queueing signal; under exhaustion it grows toward pool_timeout while
        every per-stage timer still looks healthy.
      * ``db_hold_ms``    — total time the connection was checked out.
    """
    metrics = get_metrics()
    t0 = anyio.current_time()
    session = async_session_factory()
    acquired_ms: float | None = None
    try:
        # SQLAlchemy's AsyncSession is lazy: no connection is checked out until
        # the first statement. Force it here so acquisition cost is attributed
        # to acquisition rather than smeared into the first query.
        await session.connection()
        acquired_ms = (anyio.current_time() - t0) * 1000.0
        metrics.observe(
            "db_connection_acquire_ms",
            acquired_ms,
            labels={"scope": label},
            help_text="Time waiting to check out a pooled DB connection.",
        )
        yield session
        await session.commit()
    except SQLTimeoutError as exc:
        # Pool exhausted. Previously this propagated as a bare SQLAlchemy error,
        # matched no handler, and became an unlogged 500 — the failure mode was
        # invisible in both the logs and the response body.
        waited_ms = (anyio.current_time() - t0) * 1000.0
        metrics.counter(
            "db_pool_timeouts_total",
            labels={"scope": label},
            help_text="Requests that failed waiting for a DB connection.",
        )
        logger.error(
            "DB pool exhausted in scope=%s after %.0fms — pool=%s",
            label,
            waited_ms,
            pool_stats(),
            exc_info=exc,
        )
        raise DatabaseUnavailableError() from exc
    except Exception:
        await session.rollback()
        raise
    finally:
        # close() returns the connection to the pool. In `finally` so it runs on
        # the success path, on rollback, and on cancellation alike — a leaked
        # connection here would be permanent for the process's lifetime.
        await session.close()
        if acquired_ms is not None:
            metrics.observe(
                "db_connection_hold_ms",
                (anyio.current_time() - t0) * 1000.0 - acquired_ms,
                labels={"scope": label},
                help_text="Time a DB connection stayed checked out.",
            )


def _alembic_config():
    """Build an Alembic config pointed at this project's ``alembic/`` dir."""
    from alembic.config import Config

    root = Path(__file__).resolve().parents[3]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", _settings.sync_database_url)
    # Stop alembic/env.py from calling fileConfig(), which would disable every
    # existing logger and drop the handlers setup_logging() installed. Without
    # this the app goes silent from the first migration onward.
    config.attributes["configure_logging"] = False
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
