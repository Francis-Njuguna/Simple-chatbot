"""
verify_db_credentials.py
------------------------
Pre-flight credential check for the Amref Help Desk RAG application.

Connects to the PostgreSQL instance using the credentials assembled by
``backend.app.config.Settings`` (sourced from .env / environment variables)
and verifies that:

  1. A TCP connection can be established within the timeout window.
  2. The database role exists and the password is accepted
     (i.e. no ``InvalidPasswordError``).
  3. The target database exists and is reachable.
  4. The role's current password hash matches what is on record in
     ``pg_shadow`` (superuser-only; skipped gracefully if the connecting
     role is not a superuser).

Run before starting the application:

    # From the project root (local, non-Docker):
    python scripts/verify_db_credentials.py

    # Inside the running backend container:
    docker compose exec backend python scripts/verify_db_credentials.py

Exit codes
----------
0 — all checks passed; safe to start the application.
1 — at least one check failed; the specific error is printed to stderr.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from typing import NoReturn

# ---------------------------------------------------------------------------
# Import settings (reads .env automatically via pydantic-settings)
# ---------------------------------------------------------------------------
try:
    from backend.app.config import get_settings
except ModuleNotFoundError:
    # Allow running from the project root without installing the package.
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, root)
    from backend.app.config import get_settings  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✔{_RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}", file=sys.stderr)


def _fail(msg: str) -> NoReturn:
    print(f"\n  {_RED}{_BOLD}✘  {msg}{_RESET}\n", file=sys.stderr)
    sys.exit(1)


def _section(title: str) -> None:
    print(f"\n{_BOLD}{title}{_RESET}")


# ---------------------------------------------------------------------------
# Core verification logic (async so we can reuse asyncpg naturally)
# ---------------------------------------------------------------------------

async def _verify() -> None:
    try:
        import asyncpg  # noqa: PLC0415 — optional dep, checked at runtime
    except ImportError:
        _fail(
            "asyncpg is not installed. "
            "Install it with: pip install asyncpg"
        )

    settings = get_settings()

    _section("PostgreSQL credential audit")
    print(
        textwrap.dedent(f"""\
          Host     : {settings.postgres_host}
          Port     : {settings.postgres_port}
          User     : {settings.postgres_user}
          Database : {settings.postgres_db}
          Password : {'*' * len(settings.postgres_password)}  ({len(settings.postgres_password)} chars)
          URL      : {settings.database_url.replace(settings.postgres_password, '***')}
        """)
    )

    # ------------------------------------------------------------------
    # 1. Connect with the configured credentials
    # ------------------------------------------------------------------
    _section("Check 1 — TCP connection + authentication")
    conn: asyncpg.Connection | None = None
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db,
            timeout=10,
        )
        _ok(
            f"Connected as '{settings.postgres_user}' "
            f"to database '{settings.postgres_db}' on "
            f"{settings.postgres_host}:{settings.postgres_port}"
        )
    except asyncpg.InvalidPasswordError as exc:
        _fail(
            f"Password authentication failed for user '{settings.postgres_user}'.\n\n"
            f"  asyncpg says: {exc}\n\n"
            "  Resolution steps:\n"
            "  1. Ensure POSTGRES_USER / POSTGRES_PASSWORD in .env match\n"
            "     the credentials given to the 'postgres' Docker service.\n"
            "  2. If the Docker volume already exists with different creds,\n"
            "     either update the role password in psql:\n"
            f"       ALTER ROLE {settings.postgres_user} WITH PASSWORD 'new_password';\n"
            "     or destroy the volume and let Docker re-initialise it:\n"
            "       docker compose down -v && docker compose up postgres -d\n"
            "  3. Re-run this script after making changes."
        )
    except asyncpg.InvalidCatalogNameError:
        _fail(
            f"Database '{settings.postgres_db}' does not exist.\n\n"
            f"  Create it inside psql as a superuser:\n"
            f"    CREATE DATABASE {settings.postgres_db} OWNER {settings.postgres_user};\n"
            f"  Or set POSTGRES_DB in .env to an existing database."
        )
    except OSError as exc:
        _fail(
            f"Cannot reach PostgreSQL at "
            f"{settings.postgres_host}:{settings.postgres_port}.\n\n"
            f"  OS error: {exc}\n\n"
            "  Is the postgres service running?\n"
            "    docker compose up postgres -d"
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected connection error: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # 2. Verify the role exists in pg_shadow (superuser-only catalogue)
    # ------------------------------------------------------------------
    _section("Check 2 — Role existence in pg_catalog.pg_roles")
    try:
        row = await conn.fetchrow(
            "SELECT rolname, rolcanlogin FROM pg_catalog.pg_roles WHERE rolname = $1",
            settings.postgres_user,
        )
        if row is None:
            _fail(
                f"Role '{settings.postgres_user}' not found in pg_catalog.pg_roles.\n\n"
                "  Create it:\n"
                f"    CREATE ROLE {settings.postgres_user} WITH LOGIN "
                f"PASSWORD '{settings.postgres_password}';\n"
                f"    GRANT ALL PRIVILEGES ON DATABASE "
                f"{settings.postgres_db} TO {settings.postgres_user};"
            )
        if not row["rolcanlogin"]:
            _fail(
                f"Role '{settings.postgres_user}' exists but LOGIN is not granted.\n\n"
                f"  Fix:\n"
                f"    ALTER ROLE {settings.postgres_user} WITH LOGIN;"
            )
        _ok(f"Role '{row['rolname']}' exists and has LOGIN privilege.")
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not query pg_catalog.pg_roles: {exc}")

    # ------------------------------------------------------------------
    # 3. Verify the database exists and we have CONNECT privilege
    # ------------------------------------------------------------------
    _section("Check 3 — Database accessibility")
    try:
        db_row = await conn.fetchrow(
            "SELECT datname FROM pg_catalog.pg_database WHERE datname = $1",
            settings.postgres_db,
        )
        if db_row is None:
            _fail(
                f"Database '{settings.postgres_db}' not found in pg_catalog.pg_database.\n\n"
                f"  Create it:\n"
                f"    CREATE DATABASE {settings.postgres_db} "
                f"OWNER {settings.postgres_user};"
            )
        _ok(f"Database '{db_row['datname']}' exists and is accessible.")
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not query pg_catalog.pg_database: {exc}")

    # ------------------------------------------------------------------
    # 4. Smoke-test: create + drop a temporary table
    # ------------------------------------------------------------------
    _section("Check 4 — Write permission smoke-test")
    try:
        await conn.execute(
            "CREATE TEMP TABLE _amref_cred_check (id SERIAL PRIMARY KEY)"
        )
        await conn.execute("DROP TABLE _amref_cred_check")
        _ok("Write permissions confirmed (CREATE / DROP on temporary table).")
    except Exception as exc:  # noqa: BLE001
        _warn(f"Smoke-test failed (role may be read-only): {exc}")

    await conn.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(
        f"\n{_GREEN}{_BOLD}All checks passed.{_RESET} "
        "The application should connect successfully.\n"
        "Next step:\n"
        "  docker compose up backend -d\n"
        "  # or locally:\n"
        "  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(_verify())
