"""Transport-level repair for AgentRouter's malformed SSE frames.

The problem
-----------
AgentRouter emits a literal ``data: null`` frame in the middle of an otherwise
well-formed ``chat.completion.chunk`` stream. Observed on a real request
(``claude-opus-5``, 4953-char prompt), one arrives at frame 005 — *before any
content* — and another at frame 097, near the end::

    [004] data: {... "delta":{"reasoning_content":"\\n"} ...}
    [005] data: null
    [006] data: {... "delta":{} ...}
    [007] data: {... "delta":{"content":"Happy to wal"} ...}

``null`` is valid JSON, so the OpenAI SDK's SSE decoder parses it to ``None``
and yields it as a chunk (``openai/_streaming.py:211``). ``langchain_openai``
then calls ``chunk.model_dump()`` unconditionally
(``langchain_openai/chat_models/base.py:1881``), which raises
``AttributeError: 'NoneType' object has no attribute 'model_dump'`` and kills
the stream.

Why fix it here
---------------
The frame is a protocol defect, so it is repaired at the protocol layer: both
SDKs then see a stream that is simply well-formed. The alternative — catching
the ``AttributeError`` downstream — cannot work, because that exception is
indistinguishable from the end of a stream. A null frame arriving before any
content aborts the answer entirely, and one arriving mid-answer would silently
truncate it. Neither is recoverable once the framing is already lost.

Dropping the frame is safe: a ``data: null`` chunk carries no choices, no delta
and no usage, so no information is discarded.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

_NULL_FRAME = b"data: null"


class _NullFrameFilter(httpx.AsyncByteStream):
    """Wrap a response byte stream, dropping ``data: null`` frames.

    Filtering is line-based over a buffer because a frame can be split across
    two network chunks — matching on whole chunks would miss those. When the
    ``data: null`` line is dropped, the blank line that terminates its frame is
    dropped with it, so the bytes downstream are byte-for-byte what a stream
    without the frame would have looked like.
    """

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self._stream = stream
        self._buf = b""
        # True once a `data: null` line was dropped and its terminating blank
        # line has not been dropped yet.
        self._drop_blank = False

    def _consume(self, more: bytes, *, final: bool = False) -> bytes:
        self._buf += more
        out = bytearray()
        while True:
            idx = self._buf.find(b"\n")
            if idx == -1:
                break
            line, self._buf = self._buf[: idx + 1], self._buf[idx + 1 :]
            stripped = line.strip()
            if stripped == _NULL_FRAME:
                self._drop_blank = True
                continue
            if self._drop_blank:
                self._drop_blank = False
                if not stripped:
                    continue
            out += line
        if final and self._buf:
            if self._buf.strip() != _NULL_FRAME:
                out += self._buf
            self._buf = b""
        return bytes(out)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            repaired = self._consume(chunk)
            if repaired:
                # Only yield when a whole line survived: withholding a partial
                # line costs nothing (the SDK buffers to frame boundaries
                # anyway) and keeps this from emitting a truncated frame.
                yield repaired
        tail = self._consume(b"", final=True)
        if tail:
            yield tail

    async def aclose(self) -> None:
        await self._stream.aclose()


class SSERepairTransport(httpx.AsyncBaseTransport):
    """Delegating transport that repairs streamed SSE bodies.

    Non-streaming responses are untouched: only bodies whose content type is
    ``text/event-stream`` are wrapped, so ordinary JSON responses — including
    error bodies — pass through byte-for-byte.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if "text/event-stream" not in response.headers.get("content-type", ""):
            return response
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_NullFrameFilter(response.stream),  # type: ignore[arg-type]
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def build_repaired_async_client(
    *, timeout: float, max_connections: int, max_keepalive: int
) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that transparently drops null SSE frames.

    Limits are passed in rather than left to httpx's defaults (10 connections,
    5 keep-alive) because supplying a client to ``ChatOpenAI`` overrides the
    pool the OpenAI SDK would have built (1000 / 100). Silently shrinking the
    pool to 10 would serialize concurrent requests at the HTTP layer — the
    exact failure mode this project is measuring.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
        ),
        transport=SSERepairTransport(
            httpx.AsyncHTTPTransport(
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive,
                )
            )
        ),
    )
