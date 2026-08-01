"""Amref Help Desk RAG backend application."""

import sys

# ---------------------------------------------------------------------------
# Force UTF-8 on the console streams before anything else in the package runs.
#
# On Windows, sys.stdout defaults to the legacy ANSI code page (cp1252 here),
# which cannot encode the arrows and em-dashes used throughout our log and
# print strings. config.log_db_config() runs at *import* time via
# database/session.py, so a single "→" there raised UnicodeEncodeError and
# killed uvicorn during startup before the app object was even built.
#
# This package __init__ is imported before any submodule, so reconfiguring here
# covers every entrypoint (uvicorn, scripts/ingest.py, scripts/inspect_kb.py)
# without having to strip non-ASCII from ~56 message strings. errors="replace"
# is a belt-and-braces guard for any terminal that still cannot represent a
# character — a mangled glyph is always better than a crashed process.
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:  # None under pytest capture / some WSGI hosts
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Detached or already-closed stream — nothing sensible to do.
            pass
