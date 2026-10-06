"""The streaming client against an in-process transport: arrival policies, outcomes, SSE."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from tensward.client import (
    UNLIMITED_CONNECTIONS,
    HttpxTransport,
    RequestPlan,
    StreamError,
    build_request,
    parse_sse,
    run_workload,
    steady_lead,
)
from tensward.workload import (
    ArrivalSpec,
    ChatMessage,
    ChatRequest,
    ChatTool,
    ToolFunction,
    WorkloadSpec,
)

DONE = b"data: [DONE]\n\n"


def asynchronous(test: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """Run an ``async def`` test on its own event loop (no plugin needed)."""

    @functools.wraps(test)
    def run_test(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs))

    return run_test


def frame(**body: Any) -> bytes:
    return b"data: " + json.dumps(body).encode() + b"\n\n"


CONTENT = frame(choices=[{"text": "hi"}])
USAGE = frame(choices=[], usage={"completion_tokens": 7})


class Response:
    def __init__(self, status_code: int, chunks: list[bytes], pause_s: float) -> None:
        self.status_code = status_code
        self._chunks = chunks
        self._pause_s = pause_s

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(self._pause_s)
            yield chunk


class FakeServer:
    """Answers every request after ``pause_s``, and counts how many are in flight at once."""

    def __init__(self, pause_s: float = 0.02, script: dict[int, Response] | None = None) -> None:
        self.pause_s = pause_s
        self.script = script or {}
        self.in_flight = 0
        self.peak = 0
        self.plans: list[RequestPlan] = []

    def stream(self, plan: RequestPlan) -> Any:
        return self._stream(plan)

    @asynccontextmanager
    async def _stream(self, plan: RequestPlan) -> AsyncIterator[Response]:
        self.plans.append(plan)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            index = int(plan.request_id.rsplit(":", 1)[1])
            yield self.script.get(index) or Response(200, [CONTENT, USAGE, DONE], self.pause_s)
        finally:
            self.in_flight -= 1


def workload(arrival: dict[str, Any], count: int = 9, **changes: Any) -> WorkloadSpec:
    fields = dict(
        prompts=("a prompt",),
        output_tokens=8,
        request_count=count,
        request_timeout_s=5.0,
        arrival=ArrivalSpec(**arrival),
        temperature=0.0,
        top_p=1.0,
    )
    return WorkloadSpec(**{**fields, **changes})


async def run(spec: WorkloadSpec, server: FakeServer) -> Any:
    return await run_workload(spec, run_id="r", model="m", transport=server)


@asynchronous
async def test_logprobs_follow_the_shape_of_the_api() -> None:
    arrival = {"kind": "closed_loop", "concurrency": 1}
    plain = await build_request(workload(arrival, logprobs=5), 0, run_id="r", model="m")
    assert plain.payload["logprobs"] == 5 and "top_logprobs" not in plain.payload
    chat = ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
    spec = workload(arrival, api="chat", prompts=(), chats=(chat,), logprobs=5)
    sent = (await build_request(spec, 0, run_id="r", model="m")).payload
    assert sent["logprobs"] is True and sent["top_logprobs"] == 5


@asynchronous
async def test_closed_loop_keeps_exactly_its_concurrency_in_flight() -> None:
    server = FakeServer()

    records = await run(workload({"kind": "closed_loop", "concurrency": 3}), server)

    assert server.peak == 3
    assert [r.request_id for r in records] == [f"r:{i:06d}" for i in range(9)]
    assert {r.outcome for r in records} == {"success"}
    assert {r.committed_output_tokens for r in records} == {7}
    assert all(r.dispatch_ns < r.first_content_ns < r.terminal_ns for r in records)


@asynchronous
async def test_a_lead_wave_runs_first_and_signals_when_it_has_finished() -> None:
    # The lead continues the prompt cycle at index request_count (9 here), so script 9 fails
    # the first lead request without touching the declared ones (indices 0-8).
    server = FakeServer(script={9: Response(500, [], 0.0)})
    calls: list[int] = []
    closed = workload({"kind": "closed_loop", "concurrency": 3})

    records = await run_workload(
        closed, run_id="r", model="m", transport=server, lead=3,
        on_lead_done=lambda: calls.append(1),
    )  # fmt: skip

    assert [r.request_id for r in records[:3]] == [f"r-lead:{i:06d}" for i in (9, 10, 11)]
    assert records[0].outcome == "error" and calls == [1]
    assert [r.request_id for r in records[3:]] == [f"r:{i:06d}" for i in range(9)]
    capped = workload({"kind": "capped", "rate_rps": 50.0, "max_inflight": 2})
    assert steady_lead(capped) == 0 and steady_lead(closed) == 3
    assert steady_lead(workload({"kind": "closed_loop", "concurrency": 5})) == 0  # 9 < 2 x 5


@asynchronous
async def test_capped_arrivals_follow_the_rate_but_never_exceed_the_cap() -> None:
    server = FakeServer(pause_s=0.03)
    spec = workload({"kind": "capped", "rate_rps": 200.0, "max_inflight": 2}, count=8)

    records = await run(spec, server)

    assert server.peak == 2
    gaps = {b.scheduled_ns - a.scheduled_ns for a, b in zip(records, records[1:])}
    assert gaps == {5_000_000}  # the schedule is the rate, however slow the server is
    assert records[-1].dispatch_ns - records[-1].scheduled_ns > 20_000_000  # and it waited


@asynchronous
async def test_open_loop_offers_at_the_rate_whatever_the_server_does() -> None:
    server = FakeServer(pause_s=0.05)

    await run(workload({"kind": "open_loop", "rate_rps": 200.0}, count=8), server)

    assert server.peak > 3


@asynchronous
async def test_a_failing_request_is_an_outcome_and_the_rest_carry_on() -> None:
    server = FakeServer(
        script={
            1: Response(500, [b"oops"], 0),
            2: Response(200, [CONTENT, USAGE], 0),  # ends before [DONE]
            3: Response(200, [CONTENT], 10.0),  # never finishes
        }
    )
    spec = workload({"kind": "closed_loop", "concurrency": 2}, count=5, request_timeout_s=0.2)

    records = await run(spec, server)

    assert [r.outcome for r in records] == ["success", "error", "error", "timeout", "success"]
    assert [r.committed_output_tokens for r in records] == [7, None, None, None, 7]


@asynchronous
async def test_cancelling_the_run_cancels_the_requests_in_flight() -> None:
    server = FakeServer(pause_s=10.0)
    task = asyncio.create_task(run(workload({"kind": "closed_loop", "concurrency": 2}), server))
    while server.in_flight < 2:
        await asyncio.sleep(0.001)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert server.in_flight == 0


@asynchronous
async def test_a_role_chunk_is_not_content_but_a_tool_call_is() -> None:
    role = frame(choices=[{"delta": {"role": "assistant"}}])
    call = frame(choices=[{"delta": {"tool_calls": [{"index": 0}]}}])
    server = FakeServer(script={0: Response(200, [role, call, USAGE, DONE], 0.05)})
    tool = ChatTool(type="function", function=ToolFunction(name="f", parameters={"type": "object"}))
    chat = ChatRequest(
        messages=(ChatMessage(role="user", content="hi"),), tools=(tool,), tool_choice="auto"
    )
    spec = workload(
        {"kind": "closed_loop", "concurrency": 1}, count=1, api="chat", prompts=(), chats=(chat,)
    )

    (record,) = await run(spec, server)

    assert record.first_content_ns - record.dispatch_ns >= 90_000_000  # after the second chunk
    payload = server.plans[0].payload
    assert server.plans[0].path == "/v1/chat/completions"
    assert payload["tools"][0]["function"]["name"] == "f" and payload["tool_choice"] == "auto"
    assert payload["stream"] is True and payload["max_tokens"] == 8


async def collect(chunks: list[bytes]) -> list[dict[str, Any]]:
    async def source() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    return [item async for item in parse_sse(source())]


@asynchronous
async def test_sse_frames_are_reassembled_whatever_the_chunking() -> None:
    stream = b': keep-alive\n\ndata: {"a": 1}\r\n\ndata:{"b": 2}\n\ndata: [DONE]'
    one_byte_chunks = [stream[i : i + 1] for i in range(len(stream))]

    assert await collect([stream]) == await collect(one_byte_chunks) == [{"a": 1}, {"b": 2}]


@pytest.mark.parametrize(
    "chunks",
    [
        [b'data: {"a": 1}\n\n'],  # no [DONE]: the generation may have been cut off
        [b"data: not json\n\n"],
        [b"data: [1, 2]\n\n"],
        [b"data: " + b"x" * 2_000_000],  # a line that never ends
    ],
)
@asynchronous
async def test_a_stream_that_is_not_a_finished_sse_stream_is_an_error(chunks: list[bytes]) -> None:
    with pytest.raises(StreamError):
        await collect(chunks)


@asynchronous
async def test_real_transport_reaches_more_than_100_in_flight_requests() -> None:
    """The workload client must not cap concurrency (httpx's default pool is 100)."""
    in_flight = peak = 0
    release = asyncio.Event()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal in_flight, peak
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError:
            return writer.close()
        length = int(head.lower().split(b"content-length: ")[1].split(b"\r\n")[0])
        await reader.readexactly(length)
        in_flight += 1
        peak = max(peak, in_flight)
        if peak >= 256:
            release.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(release.wait(), timeout=2.0)
        in_flight -= 1
        body = CONTENT + USAGE + DONE
        writer.write(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
            b"connection: close\r\ncontent-length: %d\r\n\r\n%s" % (len(body), body)
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0, backlog=1024)
    port = server.sockets[0].getsockname()[1]
    async with server, httpx.AsyncClient(limits=UNLIMITED_CONNECTIONS) as client:
        transport = HttpxTransport(client=client, base_url=f"http://127.0.0.1:{port}")
        spec = workload({"kind": "closed_loop", "concurrency": 256}, count=256)
        records = await asyncio.wait_for(
            run_workload(spec, run_id="r", model="m", transport=transport), timeout=20
        )

    assert peak == 256
    assert {r.outcome for r in records} == {"success"}


@asynchronous
async def test_capturing_the_response_never_lands_inside_the_measured_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensward import capture

    decode = capture._decode_capture

    async def slow_decode(data: bytes, status: int | None) -> Any:
        time.sleep(0.3)  # a blocking parse, like a 2k-token response on a busy loop
        return await decode(data, status)

    monkeypatch.setattr(capture, "_decode_capture", slow_decode)
    transport = capture.CapturingTransport(FakeServer(pause_s=0.01))

    records = await run_workload(
        workload({"kind": "closed_loop", "concurrency": 1}, count=3),
        run_id="r",
        model="m",
        transport=transport,
    )

    assert all(r.terminal_ns - r.dispatch_ns < 150_000_000 for r in records)  # 3 x 0.03 s
    assert transport.responses == {}  # nothing decoded while the run was timed
    await transport.decode_all()
    assert [transport.responses[r.request_id].text for r in records] == ["hi"] * 3
