"""Import + wiring smoke check for the DB connection-lifecycle refactor.

Verifies the pieces that only fail at import/wire time (circular imports, stale
call signatures, unregistered handlers) so the load test is not the thing that
discovers them.
"""

import inspect
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.app.main as m
from backend.app.api.dependencies import get_rag_service
from backend.app.core.request_context import get_request_id, sanitize_request_id
from backend.app.database.session import db_scope, pool_stats, publish_pool_gauges
from backend.app.services.rag_service import RAGService
from backend.app.utils.metrics import get_metrics

print("main imported OK")
print("pool_stats:", pool_stats())
print("get_rag_service sig:", inspect.signature(get_rag_service))
print("RAGService sig:", inspect.signature(RAGService.__init__))
print("db_scope sig:", inspect.signature(db_scope))

handlers = sorted(getattr(k, "__name__", str(k)) for k in m.app.exception_handlers)
print("exception handlers:", handlers)

mw = [getattr(x.cls, "__name__", str(x.cls)) for x in m.app.user_middleware]
print("middleware (outermost first):", mw)

publish_pool_gauges()
gauges = {k: v for k, v in get_metrics().snapshot()["gauges"].items() if "db_pool" in k}
print("pool gauges:", gauges)

print("request id outside request:", get_request_id())
print("sanitize(bad):", sanitize_request_id("evil\nline: INJECTED"))
print("sanitize(good):", sanitize_request_id("abc-123_x.y"))
print("OK")
