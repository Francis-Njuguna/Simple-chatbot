"""Alembic migration environment.

Credential resolution
---------------------
The URL is taken from ``Settings.sync_database_url`` (psycopg2 dialect) rather
than from ``alembic.ini``. That keeps a single source of truth: the app, the
ingestion CLI and migrations all honour the same ``DATABASE_URL`` /
``POSTGRES_*`` precedence documented in ``backend/app/config.py``.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Allow `alembic` to run from the project root without the package installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings  # noqa: E402
from backend.app.database.models import Base  # noqa: E402

config = context.config

# ``fileConfig`` defaults to ``disable_existing_loggers=True`` and replaces the
# root handler set. Run standalone that is what you want; run *inside the app*
# (lifespan → init_db → command.upgrade) it silently tears down the handlers
# ``setup_logging()`` just installed and disables every logger that already
# exists — including ``uvicorn.error``. The symptom is brutal to diagnose: the
# app logs normally until the first migration, then goes permanently silent, so
# unhandled 500s vanish from app.log AND the console. ``_alembic_config()`` in
# database/session.py sets this flag to keep migrations from touching logging.
if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name)

# Autogenerate compares against the ORM metadata.
target_metadata = Base.metadata


def _database_url() -> str:
    """Sync (psycopg2) URL — Alembic does not use the asyncpg driver."""
    return get_settings().sync_database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade --sql``)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations in a transaction."""
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
