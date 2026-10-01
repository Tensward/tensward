"""The streaming request client: offers a workload to an OpenAI-compatible server and records
what the client saw.

Arrivals follow the workload's ``ArrivalSpec``: ``closed_loop`` keeps N requests in flight,
``open_loop`` offers a fixed rate whatever the server does, and ``capped`` offers that rate
but holds requests back once a cap is in flight. Each request gets a :class:`RequestRecord` on
the client's monotonic clock. Generated text is never kept here, and a metric the server did
not report stays ``None`` rather than being guessed.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Literal, Mapping, Protocol

import httpx

from .workload import ChatRequest, WorkloadApi, WorkloadSpec

CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
HTTP_ERROR_MIN = 400
MAX_SSE_LINE_BYTES = 1_048_576
NANOS_PER_SECOND = 1_000_000_000

RequestOutcome = Literal["success", "error", "timeout"]


class StreamError(Exception):
    """A response that is not a completed SSE stream: an error status, a malformed frame, or
    a stream that ended before ``data: [DONE]``."""


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One request as the client observed it. All instants are client monotonic nanoseconds.

    ``first_content_ns`` is the first chunk that carried generated text or a tool call (a role
    or usage chunk is not content). ``committed_output_tokens`` is the server's own count from
    the final usage object, kept only for a successful request.
    """

    request_id: str
    scheduled_ns: int
    dispatch_ns: int
    first_content_ns: int | None
    terminal_ns: int
    outcome: RequestOutcome
    committed_output_tokens: int | None


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """One fully built request: where it goes and its exact JSON payload."""

    request_id: str
    api: WorkloadApi
    path: str
    payload: Mapping[str, Any]
    timeout_s: float


class ResponseStream(Protocol):
    """One streaming HTTP response, as the client reads it."""

    @property
    def status_code(self) -> int: ...

    def aiter_bytes(self) -> AsyncIterator[bytes]: ...


class Transport(Protocol):
    """How one request is offered and its response streamed back."""

    def stream(self, plan: RequestPlan) -> AbstractAsyncContextManager[ResponseStream]: ...


# httpx's default pool holds 100 connections, which would silently cap any declared concurrency
# above that (and bill the wait for a free connection to the request's TTFT).
UNLIMITED_CONNECTIONS = httpx.Limits(max_connections=None, max_keepalive_connections=None)


class HttpxTransport:
    """The real transport: one streaming POST per request. It neither follows redirects nor
    retries, because a retry would be a second attempt nobody declared. The client it is given
    must use ``UNLIMITED_CONNECTIONS``: with it a request's connection is opened (or reused)
    without queueing, so ``dispatch_ns`` is when the request is actually offered."""

    def __init__(self, *, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    def stream(self, plan: RequestPlan) -> AbstractAsyncContextManager[ResponseStream]:
        return self._client.stream(
            "POST", self._base_url + plan.path, json=dict(plan.payload), timeout=plan.timeout_s
        )


# --- SSE ---------------------------------------------------------------------------------


def _data(line: bytes) -> str | None:
    """The payload of an SSE ``data:`` line; comments and other fields are ignored."""
    text = line.rstrip(b"\r").decode("utf-8", "replace")
    if not text.startswith("data:"):
        return None
    return text.removeprefix("data:").removeprefix(" ")


async def parse_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Yield each JSON frame of an SSE byte stream up to ``data: [DONE]``.

    Chunk boundaries are irrelevant. A stream that ends without the sentinel is an error: the
    sentinel is the evidence that generation finished rather than merely stopped.
    """
    buffer = b""
    async for chunk in chunks:
        *lines, buffer = (buffer + chunk).split(b"\n")
        for line in lines:
            payload = _data(line)
            if payload == "[DONE]":
                return
            if payload is not None:
                yield _frame(payload)
        if len(buffer) > MAX_SSE_LINE_BYTES:
            raise StreamError("an SSE line is too long")
    if _data(buffer) != "[DONE]":
        raise StreamError("the stream ended before [DONE]")


def _frame(payload: str) -> dict[str, Any]:
    try:
        frame = json.loads(payload)
    except (ValueError, RecursionError):
        raise StreamError("a frame is not valid JSON") from None
    if not isinstance(frame, dict):
        raise StreamError("a frame is not a JSON object")
    return frame


def _carries_content(frame: Mapping[str, Any], api: WorkloadApi) -> bool:
    """Whether a frame carries generated text, reasoning or a tool call."""
    for choice in frame.get("choices") or ():
        if api == "chat":
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                return True
        elif choice.get("text"):
            return True
    return False


def _committed_tokens(frame: Mapping[str, Any]) -> int | None:
    """The completion token count of a final usage object, when the server sent one."""
    usage = frame.get("usage")
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None


# --- requests ------------------------------------------------------------------------------


def chat_body(chat: ChatRequest) -> dict[str, Any]:
    """The OpenAI ``messages`` (and ``tools``, when offered) of one chat prompt."""
    body: dict[str, Any] = {"messages": [message.model_dump() for message in chat.messages]}
    if chat.tools:
        body["tools"] = [tool.model_dump(exclude_none=True) for tool in chat.tools]
    return body


def build_request(workload: WorkloadSpec, index: int, *, run_id: str, model: str) -> RequestPlan:
    """The request for arrival ``index``: prompts are offered in order and cycle."""
    max_tokens = workload.output_tokens
    if workload.api == "chat":
        chat = workload.chats[index % len(workload.chats)]
        path, body = CHAT_PATH, chat_body(chat)
        if chat.tool_choice is not None:
            body["tool_choice"] = chat.tool_choice
        max_tokens = chat.max_tokens or max_tokens
    else:
        path, body = COMPLETIONS_PATH, {"prompt": workload.prompts[index % len(workload.prompts)]}
    body.update(
        model=model,
        stream=True,
        stream_options={"include_usage": True},
        n=1,
        max_tokens=max_tokens,
        temperature=workload.temperature,
        top_p=workload.top_p,
    )
    optional = {
        "seed": workload.seed,
        "stop": list(workload.stop) or None,
        "logprobs": workload.logprobs,
        "structured_outputs": workload.structured_output,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return RequestPlan(
        f"{run_id}:{index:06d}", workload.api, path, body, workload.request_timeout_s
    )


async def _attempt(plan: RequestPlan, scheduled_ns: int, transport: Transport) -> RequestRecord:
    """Offer one request and describe it. A timeout or a transport or protocol failure is that
    request's outcome; anything else is a bug and propagates."""
    dispatch_ns = time.monotonic_ns()
    first_content_ns = tokens = None
    outcome: RequestOutcome = "success"
    try:
        async with asyncio.timeout(plan.timeout_s), transport.stream(plan) as response:
            if response.status_code >= HTTP_ERROR_MIN:
                raise StreamError(f"the server answered with status {response.status_code}")
            async for frame in parse_sse(response.aiter_bytes()):
                if first_content_ns is None and _carries_content(frame, plan.api):
                    first_content_ns = time.monotonic_ns()
                if (counted := _committed_tokens(frame)) is not None:
                    tokens = counted
    except (TimeoutError, httpx.TimeoutException):
        outcome = "timeout"
    except (httpx.HTTPError, StreamError):
        outcome = "error"
    return RequestRecord(
        request_id=plan.request_id,
        scheduled_ns=scheduled_ns,
        dispatch_ns=dispatch_ns,
        first_content_ns=first_content_ns,
        terminal_ns=time.monotonic_ns(),
        outcome=outcome,
        committed_output_tokens=tokens if outcome == "success" else None,
    )


async def run_workload(
    workload: WorkloadSpec,
    *,
    run_id: str,
    model: str,
    transport: Transport,
    on_done: Callable[[int], None] = lambda count: None,
) -> tuple[RequestRecord, ...]:
    """Offer ``workload.request_count`` requests as its arrival policy says; return one record
    per request, in order. ``on_done`` hears how many requests have finished so far. If the
    caller is cancelled, every request in flight is too."""
    finished = 0
    arrival = workload.arrival
    start_ns = time.monotonic_ns()
    limit = arrival.concurrency if arrival.kind == "closed_loop" else arrival.max_inflight
    slots = asyncio.Semaphore(limit) if limit else None

    async def offer(index: int, scheduled_ns: int) -> RequestRecord:
        nonlocal finished
        try:
            plan = build_request(workload, index, run_id=run_id, model=model)
            record = await _attempt(plan, scheduled_ns, transport)
            finished += 1
            on_done(finished)
            return record
        finally:
            if slots:
                slots.release()

    async with asyncio.TaskGroup() as group:
        tasks = []
        for index in range(workload.request_count):
            due_ns = None  # a rate schedule follows the rate alone, never the server
            if arrival.rate_rps:
                due_ns = start_ns + round(index * NANOS_PER_SECOND / arrival.rate_rps)
                await asyncio.sleep(max(0, due_ns - time.monotonic_ns()) / NANOS_PER_SECOND)
            if slots:
                await slots.acquire()
            scheduled_ns = time.monotonic_ns() if due_ns is None else due_ns
            tasks.append(group.create_task(offer(index, scheduled_ns)))
    return tuple(task.result() for task in tasks)
