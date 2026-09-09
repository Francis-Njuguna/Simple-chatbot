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

Why the stream must be uncompressed
-----------------------------------
A transport sees the response body *before* content decoding — ``response.stream``
at this layer is still whatever the wire carried. AgentRouter negotiates Brotli
(``content-encoding: br``) when a client advertises it, and httpx advertises it
by default, so the filter was handed compressed bytes and could never match a
frame; httpx then decompressed downstream and passed the intact ``data: null``
to the SDK. Verified against the live gateway: with httpx's default
``Accept-Encoding`` the raw stream contains 0 readable ``data:`` lines, and with
``identity`` it contains ~198 lines including 2 real null frames.

So the transport pins ``Accept-Encoding: identity`` on the requests it filters.
That makes this module's central assumption — that it is reading SSE text — true,
rather than leaving it as an unstated precondition. Decompressing here instead
would mean re-implementing httpx's content negotiation and stripping the header
so it does not decode twice; asking for plaintext is the smaller contract. The
cost is giving up compression on the LLM response body, which for a token stream
is a poor trade anyway: frames are tiny and latency-critical, and Brotli buffers
to fill its window.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

_NULL_FRAME = b"data: null"

# Only these bodies are filtered, and only for them is compression refused.
_SSE_ROUTE_HINT = "/chat/completions"


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

    Requests are sent with ``Accept-Encoding: identity`` so the body this
    transport inspects is SSE text rather than a compressed blob. See the module
    docstring — without it the filter silently matches nothing.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Refuse compression only on the completions route. Scoping it keeps any
        # other traffic sharing this client (model lists, health probes) on
        # normal content negotiation.
        if _SSE_ROUTE_HINT in request.url.path:
            request.headers["Accept-Encoding"] = "identity"

        response = await self._inner.handle_async_request(request)
        if "text/event-stream" not in response.headers.get("content-type", ""):
            return response

        # If a gateway ignores `identity` and compresses anyway, the filter
        # cannot read the frames. Passing the body through unfiltered keeps the
        # stream working (httpx still decodes it) and the null frame will crash
        # as it did before, which is strictly better than corrupting the bytes.
        encoding = response.headers.get("content-encoding", "").strip().lower()
        if encoding and encoding != "identity":
            logger.error(
                "SSE stream arrived with content-encoding=%r despite requesting "
                "identity; null-frame repair is INACTIVE for this response.",
                encoding,
            )
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
    *, timeout: float | httpx.Timeout, max_connections: int, max_keepalive: int
) -> httpx.AsyncClient:
    """An ``httpx.AsyncClient`` that transparently drops null SSE frames.

    ``timeout`` accepts an ``httpx.Timeout`` so callers can budget connect and
    read separately — reaching an unreachable host and waiting on a slow
    generation are different failures on different timescales. Note that
    whatever is passed here is only honoured if the OpenAI SDK is given no
    timeout of its own; see the note at the ``ChatOpenAI`` call in ``rag.llm``.

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
