"""Keep what each request generated, without touching the client: a transport wrapper that
tees the response bytes and decodes them afterwards into text, tool calls and error messages."""

from __future__ import annotations

import json
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .client import HTTP_ERROR_MIN, RequestPlan, ResponseStream, Transport, parse_sse
from .toolcalls import ToolCall

MAX_ERROR_BODY_BYTES = 4096
MAX_ERROR_CHARS = 500


@dataclass(slots=True)
class CapturedResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    http_status: int | None = None  # set only when the server answered with an error status
    error: str | None = None  # the server's own message for that error, at most 500 characters


class _TeeStream:
    def __init__(self, inner: ResponseStream, buffer: bytearray) -> None:
        self._inner = inner
        self._buffer = buffer

    @property
    def status_code(self) -> int:
        return self._inner.status_code

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner.aiter_bytes():
            self._buffer.extend(chunk)
            yield chunk


class CapturingTransport:
    """Wraps a transport and keeps each request's generated text, without touching the client.

    Only the raw bytes are kept while requests run: decoding them costs milliseconds of event
    loop per long response, which would land inside the client's timings. ``decode_all`` fills
    ``responses`` once the run is over.
    """

    def __init__(self, inner: Transport) -> None:
        self._inner = inner
        self._raw: dict[str, tuple[bytes, int | None]] = {}
        self.responses: dict[str, CapturedResponse] = {}

    async def decode_all(self) -> None:
        for request_id, (data, status) in self._raw.items():
            self.responses[request_id] = await _decode_capture(data, status)

    def stream(self, plan: RequestPlan) -> AbstractAsyncContextManager[ResponseStream]:
        return self._stream(plan)

    @asynccontextmanager
    async def _stream(self, plan: RequestPlan) -> AsyncIterator[ResponseStream]:
        buffer = bytearray()
        status: int | None = None
        try:
            async with self._inner.stream(plan) as response:
                status = response.status_code
                try:
                    yield _TeeStream(response, buffer)
                finally:
                    if status >= HTTP_ERROR_MIN and not buffer:
                        # The client refuses an error status without reading its body.
                        await _read_error_body(response, buffer)
        finally:
            self._raw[plan.request_id] = (bytes(buffer), status)


async def _read_error_body(response: ResponseStream, buffer: bytearray) -> None:
    try:
        async for chunk in response.aiter_bytes():
            buffer.extend(chunk)
            if len(buffer) >= MAX_ERROR_BODY_BYTES:
                break
    except httpx.HTTPError:
        pass  # the status alone still says why the request failed


def _error_message(body: bytes) -> str:
    """The server's message from an error body (OpenAI-style JSON, else the raw text)."""
    text = body.decode("utf-8", "replace").strip()
    try:
        payload = json.loads(text)
    except ValueError:
        return text[:MAX_ERROR_CHARS]
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    if not isinstance(message, str) and isinstance(payload, dict):
        message = payload.get("message")
    return (message if isinstance(message, str) else text)[:MAX_ERROR_CHARS]


async def _decode_capture(data: bytes, status: int | None) -> CapturedResponse:
    captured = CapturedResponse()
    if status is not None and status >= HTTP_ERROR_MIN:
        captured.http_status = status
        captured.error = _error_message(data)
        return captured

    async def chunks() -> AsyncIterator[bytes]:
        yield data

    fragments: dict[int, list[str]] = {}  # tool-call index -> [name, arguments] so far
    try:
        async for frame in parse_sse(chunks()):
            for choice in frame.get("choices") or ():
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                for piece in (choice.get("text"), delta.get("content")):
                    if isinstance(piece, str):
                        captured.text += piece
                for fragment in delta.get("tool_calls") or ():
                    function = fragment.get("function") or {}
                    call = fragments.setdefault(fragment.get("index", 0), ["", ""])
                    call[0] += function.get("name") or ""
                    call[1] += function.get("arguments") or ""
                if choice.get("finish_reason"):
                    captured.finish_reason = choice["finish_reason"]
            usage = frame.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                captured.prompt_tokens = usage["prompt_tokens"]
    except Exception:  # noqa: BLE001 - a partial stream keeps the text read so far
        pass
    captured.tool_calls = [
        ToolCall(name, arguments) for _, (name, arguments) in sorted(fragments.items())
    ]
    return captured
