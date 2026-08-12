"""Logging configuration."""

import logging
import sys
from pathlib import Path

from backend.app.config import get_settings
from backend.app.core.request_context import get_request_id


class _RequestIdFilter(logging.Filter):
    """Attach the current request id to every record.

    A filter rather than a custom Formatter: filters run for records from
    *every* logger, including third-party ones (sqlalchemy, httpx, uvicorn), so
    a pool-timeout logged by SQLAlchemy itself is still correlatable. Records
    emitted outside a request get ``-``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def setup_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Idempotent: uvicorn --reload and the test suite both call this more than
    # once per process, and duplicated handlers mean duplicated log lines.
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | [%(request_id)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    request_id_filter = _RequestIdFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    root.addHandler(console_handler)

    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_id_filter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
