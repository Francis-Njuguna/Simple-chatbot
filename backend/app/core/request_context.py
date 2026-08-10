"""Per-request correlation id.

Why this exists
---------------
When the connection pool was exhausted under load, 84 requests failed and the
logs said nothing useful — there was no way to tie a failure to the request that
caused it, or to see which stage it died in. A correlation id fixes that: it is
generated once per request, injected into *every* log line emitted while that
request is on the stack, and returned to the caller in ``X-Request-ID`` so a
client-reported failure can be found in the log without guessing at timestamps.

A ``ContextVar`` rather than a parameter threaded through call sites: asyncio
copies the context at task-creation time, so anything awaited inside the request
— including ``db_scope`` deep in the service layer — sees the right value with
no plumbing. It is also correct under concurrency in a way a module global is
not: 50 in-flight requests each read their own value.

This module is a leaf. It imports nothing from the application so any layer
(middleware, services, the database session helper) can use it without a cycle.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar

#: Header used to accept an upstream id (load balancer, gateway, frontend) and
#: to echo the resolved one back on the response.
REQUEST_ID_HEADER = "X-Request-ID"

#: ``-`` rather than ``None`` so log formatting never has to special-case work
#: that happens outside a request (startup, migrations, CLI scripts).
_request_id: ContextVar[str] = ContextVar("request_id", default="-")

# An id we did not generate is attacker-controlled input: it lands in log files
# and a response header, so cap the length and drop anything that could forge a
# log line or split a header.
_MAX_INBOUND_LEN = 64
_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def new_request_id() -> str:
    """A fresh short id. Short because it is prefixed onto every log line."""
    return uuid.uuid4().hex[:12]


def sanitize_request_id(candidate: str | None) -> str:
    """Accept an upstream id if it is safe, otherwise mint a new one."""
    if not candidate:
        return new_request_id()
    candidate = candidate.strip()[:_MAX_INBOUND_LEN]
    if not candidate or not set(candidate) <= _ALLOWED:
        return new_request_id()
    return candidate


def set_request_id(request_id: str):
    """Bind the id for the current context. Returns the reset token."""
    return _request_id.set(request_id)


def reset_request_id(token) -> None:
    _request_id.reset(token)


def get_request_id() -> str:
    """Current request's id, or ``-`` outside a request."""
    return _request_id.get()


class RequestContextMiddleware:
    """Bind a request id and a start timestamp for the duration of the request.

    A raw ASGI middleware, not ``BaseHTTPMiddleware``. The latter runs the
    downstream app in a separate task and pipes the response body through a
    memory stream, which adds a hop to every chunk of an SSE stream — exactly
    the path ``/chat/stream`` uses, where time-to-first-token is the metric that
    matters. This version wraps ``send`` and touches nothing else.

    Also records ``started`` (``perf_counter``) in the ASGI scope's state so an
    exception handler can report how long the request ran before it failed,
    without a second clock or a middleware-ordering dependency.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = None
        header_name = REQUEST_ID_HEADER.lower().encode("latin-1")
        for key, value in scope.get("headers", ()):
            if key == header_name:
                inbound = value.decode("latin-1", "replace")
                break

        request_id = sanitize_request_id(inbound)
        # ``scope["state"]`` is what Starlette's ``request.state`` reads.
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        scope["state"]["started"] = time.perf_counter()

        async def send_with_header(message) -> None:
            if message["type"] == "http.response.start":
                # Copy: the downstream response owns its header list and may be
                # reused (e.g. a cached error response instance).
                headers = list(message.get("headers", ()))
                headers.append((header_name, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        token = set_request_id(request_id)
        try:
            await self.app(scope, receive, send_with_header)
        finally:
            reset_request_id(token)


def request_elapsed_ms(scope_state: dict | None) -> float | None:
    """Milliseconds since the request started, if the middleware recorded it."""
    if not scope_state:
        return None
    started = scope_state.get("started")
    if started is None:
        return None
    return (time.perf_counter() - started) * 1000.0
