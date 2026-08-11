"""AgentRouter's malformed `data: null` SSE frames are dropped in transport.

The gateway emits ``data: null`` mid-stream. It is valid JSON, so the OpenAI
SDK parses it to ``None`` and langchain then calls ``.model_dump()`` on it,
raising ``AttributeError`` and killing the stream. Observed on a real request:
one null frame at position 005 (before any content) and another at 097.

These pin the filter's contract: null frames vanish, every other byte survives
unchanged, and the repair holds when a frame is split across network chunks.
"""

from __future__ import annotations

import httpx
import pytest

from backend.app.rag.sse_repair import SSERepairTransport, _NullFrameFilter


class _FakeStream:
    """An httpx byte stream that yields a fixed list of chunks."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


async def _filter(chunks: list[bytes]) -> bytes:
    out = bytearray()
    async for piece in _NullFrameFilter(_FakeStream(chunks)):  # type: ignore[arg-type]
        out += piece
    return bytes(out)


def _frame(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


@pytest.mark.asyncio
async def test_drops_a_null_frame_and_keeps_the_rest() -> None:
    body = _frame('{"a":1}') + _frame("null") + _frame('{"b":2}')

    assert await _filter([body]) == _frame('{"a":1}') + _frame('{"b":2}')


@pytest.mark.asyncio
async def test_stream_without_null_frames_is_byte_identical() -> None:
    """The filter must be invisible to every provider that behaves."""
    body = _frame('{"a":1}') + _frame('{"b":2}') + b"data: [DONE]\n\n"

    assert await _filter([body]) == body


@pytest.mark.asyncio
async def test_null_frame_split_across_network_chunks() -> None:
    """A frame does not have to arrive whole.

    TCP splits wherever it likes, so matching on whole chunks would let a null
    frame through whenever it straddles a boundary. The filter buffers by line
    for exactly this case.
    """
    head, null, tail = _frame('{"a":1}'), _frame("null"), _frame('{"b":2}')
    body = head + null + tail
    # Split inside the word "null" itself, the worst case for the match.
    cut = len(head) + len("data: nu")
    chunks = [body[:cut], body[cut:]]

    assert await _filter(chunks) == head + tail


@pytest.mark.asyncio
async def test_null_frame_first_does_not_swallow_the_answer() -> None:
    """The real failure: a null frame arrives before any content.

    This is why the repair is at the transport layer rather than an exception
    handler around the stream — at frame 005 there is no partial answer to
    salvage, so catching the AttributeError downstream loses everything.
    """
    body = _frame("null") + _frame('{"delta":{"content":"Happy to wal"}}')

    assert await _filter([body]) == _frame('{"delta":{"content":"Happy to wal"}}')


@pytest.mark.asyncio
async def test_trailing_null_frame_without_final_newline() -> None:
    body = _frame('{"a":1}') + b"data: null"

    assert await _filter([body]) == _frame('{"a":1}')


@pytest.mark.asyncio
async def test_content_that_merely_contains_null_is_kept() -> None:
    """Only the exact frame is dropped, never a chunk that talks about null."""
    body = _frame('{"delta":{"content":"data: null"}}')

    assert await _filter([body]) == body


@pytest.mark.asyncio
async def test_non_sse_responses_are_not_wrapped() -> None:
    """JSON bodies — including error bodies — must pass through untouched."""

    class _Inner(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                headers={"content-type": "application/json"},
                content=b'{"error":"unauthorized client detected"}',
                request=request,
            )

    transport = SSERepairTransport(_Inner())
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post("https://example.invalid/v1/chat/completions")

    assert response.status_code == 401
    assert response.content == b'{"error":"unauthorized client detected"}'


@pytest.mark.asyncio
async def test_sse_response_is_repaired_end_to_end() -> None:
    """Through a real httpx client, the way the OpenAI SDK consumes it."""
    body = _frame('{"a":1}') + _frame("null") + _frame('{"b":2}') + b"data: [DONE]\n\n"

    class _Inner(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
                request=request,
            )

    transport = SSERepairTransport(_Inner())
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream(
            "POST", "https://example.invalid/v1/chat/completions"
        ) as response:
            received = b"".join([chunk async for chunk in response.aiter_bytes()])

    assert b"data: null" not in received
    assert received == _frame('{"a":1}') + _frame('{"b":2}') + b"data: [DONE]\n\n"
