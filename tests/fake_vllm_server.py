"""A stdlib-only stand-in for ``vllm serve``, used by the end-to-end ``tensward analyse`` test.

It serves only what the collection client and the normalizer call: ``/health``,
``/v1/models``, ``/tokenize``, streaming ``/v1/completions`` (SSE with a final usage chunk)
and ``/metrics`` in the vLLM 0.30 Prometheus shape. Generated text is deterministic. Unknown
command-line flags are ignored so the real ``vllm serve`` argument vector can be passed
unchanged. As vLLM does, the API key (env ``VLLM_API_KEY``) protects ``/v1`` only.
``/metrics`` also carries the per-request queue and prefill time sums that vLLM 0.30 exports.

``/v1/chat/completions`` streams the same way. When the request offers tools and the server was
started with ``--enable-auto-tool-choice --tool-call-parser <name>`` it answers with one
deterministic tool call to the first tool, its arguments built from the schema's required keys;
without those flags it refuses, as vLLM does, with HTTP 400.

A prompt is counted as one token per word and each image as ``IMAGE_TOKENS`` tokens. An image
must be a base64 ``data:`` URL: any other URL (a path the engine would try to fetch) is refused
with HTTP 400. As vLLM does, a request whose prompt plus ``max_tokens`` exceeds
``--max-model-len`` is refused with HTTP 400 and an OpenAI-style error body.

When started with ``--profiler-config '{"profiler": "torch", "torch_profiler_dir": DIR}'`` it
serves ``POST /start_profile`` and ``/stop_profile`` like vLLM 0.30: stopping writes the
synthetic Chrome trace next to this file into DIR as ``fake_<pid>.<ms>.pt.trace.json.gz``.

Decode steps are paced on a running deadline, not slept one by one, so the time spent sending a
frame is not added to the step and the stream's TPOT follows the model above, not the machine.

Five flags change how it behaves, so a tuning loop has something to find and the trade-offs
are real (a decode step costs ``TOKEN_DELAY_S * (1 + running / CONTENTION_SEQS)``):

* ``--max-num-seqs`` caps concurrent requests; the rest wait. More of them raise throughput and
  end queueing (TTFT) but make every step slower (TPOT);
* ``--max-num-batched-tokens`` is the prefill chunk size: a prompt is prefilled in chunks, each
  costing ``CHUNK_OVERHEAD_S`` extra, and while a chunk runs every decode step is stalled by
  the chunk's compute time. Big chunks prefill faster (throughput, TTFT) but spike TPOT;
* ``--enable-prefix-caching`` makes a prompt whose first ``PREFIX_WORDS`` words were seen before
  prefill only its last ``CACHED_PREFILL_TOKENS`` tokens (less TTFT and less stall);
* ``--gpu-memory-utilization`` sets the KV capacity to ``10 * value`` concurrent sequences;
  above that the server reports preemptions and decodes at half speed;
* ``--kv-cache-dtype fp8`` makes it exit with an error, like an unsupported engine setting;
  ``fp8_e4m3`` answers prompts whose text has an odd length with ``alt0 alt1 ...``, like an engine
  whose answers drift.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SYNTHETIC_TRACE = Path(__file__).with_name("synthetic_trace.json")
# vLLM 0.30's cache_config_info line (labels and all), as a real server printed it.
CACHE_CONFIG_INFO = next(
    line
    for line in Path(__file__).with_name("vllm030_gemma4_metrics.txt").read_text().splitlines()
    if line.startswith("vllm:cache_config_info{")
)
BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0)
TOKEN_DELAY_S = 0.001
CONTENTION_SEQS = 2.0  # each extra running request adds 1/2 of a base step to every step
PREFILL_S_PER_TOKEN = 0.00002
PREFIX_WORDS = 100  # a prompt starting with words seen before is a prefix-cache hit
CACHED_PREFILL_TOKENS = 16  # ... and prefills only its last tokens
CHUNK_OVERHEAD_S = 0.0002
IMAGE_TOKENS = 256  # what vLLM counts for a Gemma 4 image, whatever its size


class Slots:
    """A first-come first-served counting semaphore (threading.Semaphore can starve waiters)."""

    def __init__(self, count: int) -> None:
        self.free = count
        self.queue: deque[object] = deque()
        self.changed = threading.Condition()

    def acquire(self) -> None:
        ticket = object()
        with self.changed:
            self.queue.append(ticket)
            self.changed.wait_for(lambda: self.free > 0 and self.queue[0] is ticket)
            self.queue.popleft()
            self.free -= 1
            self.changed.notify_all()

    def release(self) -> None:
        with self.changed:
            self.free += 1
            self.changed.notify_all()


SAMPLE_VALUES = {
    "string": "x",
    "integer": 1,
    "number": 1.5,
    "boolean": True,
    "array": [],
    "object": {},
}


def tool_call_pieces(tool: dict) -> list[dict]:
    """Streamed deltas of one call to ``tool``: the name first, then its arguments in thirds."""
    parameters = tool["function"]["parameters"]
    properties = parameters.get("properties", {})
    arguments = json.dumps(
        {
            key: SAMPLE_VALUES.get(properties.get(key, {}).get("type"), "x")
            for key in parameters.get("required", [])
        }
    )
    size = -(-len(arguments) // 3)
    parts = [arguments[i : i + size] for i in range(0, len(arguments), size)]
    first = {"index": 0, "id": "call_0", "type": "function"}
    calls = [{**first, "function": {"name": tool["function"]["name"], "arguments": ""}}]
    calls += [{"index": 0, "function": {"arguments": part}} for part in parts]
    return [{"delta": {"tool_calls": [call]}} for call in calls]


def message_text(message: dict) -> str:
    """The text of a message, whether its content is a string or a list of parts."""
    content = message.get("content", "")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if part["type"] == "text")
    return str(content)


def image_urls(request: dict) -> list[str]:
    return [
        part["image_url"]["url"]
        for message in request.get("messages", [])
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]


def valid_image_url(url: str) -> bool:
    header, _, payload = url.partition(",")
    if not header.startswith("data:image/") or not header.endswith(";base64"):
        return False
    try:
        base64.b64decode(payload, validate=True)
    except binascii.Error:
        return False
    return True


def prompt_text(request: dict) -> str:
    if "messages" in request:
        return " ".join(message_text(message) for message in request["messages"])
    return str(request.get("prompt", ""))


def chat_words(request: dict) -> int:
    """Prompt size of a chat request: one token per word of its messages and tool schemas, and
    ``IMAGE_TOKENS`` per image."""
    words = len(prompt_text(request).split()) + len(json.dumps(request.get("tools", [])).split())
    return words + IMAGE_TOKENS * len(image_urls(request))


class State:
    def __init__(
        self,
        model: str,
        api_key: str | None,
        max_seqs: int,
        kv_slots: float,
        max_model_len: int,
        batched_tokens: int = 2048,
        prefix_caching: bool = False,
        tools_enabled: bool = False,
        trace_dir: str | None = None,
        drift: bool = False,
    ) -> None:
        self.drift = drift
        self.tools_enabled = tools_enabled
        self.trace_dir = trace_dir
        self.model = model
        self.batched_tokens = batched_tokens
        self.prefix_caching = prefix_caching
        self.seen_prefixes: set[tuple[str, ...]] = set()
        self.prefilling_tokens = 0  # tokens of the prefill chunks running right now
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.slots = Slots(max_seqs)
        self.kv_slots = kv_slots
        self.lock = threading.Lock()
        self.running = 0
        self.waiting = 0
        self.preemptions = 0
        self.success = 0
        self.queue_seconds = self.prefill_seconds = 0.0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.ttft_counts = [0] * (len(BUCKETS) + 1)
        self.ttft_sum = 0.0

    def prefill(self, prompt_tokens: int, text: str) -> None:
        """Prefill in chunks; each chunk stalls the decode steps of everyone else."""
        remaining = prompt_tokens
        if self.prefix_caching:
            prefix = tuple(text.split()[:PREFIX_WORDS])
            with self.lock:
                hit = prefix in self.seen_prefixes
                self.seen_prefixes.add(prefix)
            if hit:
                remaining = min(remaining, CACHED_PREFILL_TOKENS)
        while remaining > 0:
            chunk = min(remaining, self.batched_tokens)
            with self.lock:
                self.prefilling_tokens += chunk
            time.sleep(chunk * PREFILL_S_PER_TOKEN + CHUNK_OVERHEAD_S)
            with self.lock:
                self.prefilling_tokens -= chunk
            remaining -= chunk

    def step_seconds(self) -> float:
        """One decode step: slower with more running requests, thrashing and running prefills."""
        thrashing = 2 if self.running > self.kv_slots else 1
        contention = 1 + self.running / CONTENTION_SEQS
        return TOKEN_DELAY_S * thrashing * contention + self.prefilling_tokens * PREFILL_S_PER_TOKEN

    def observe_ttft(self, seconds: float) -> None:
        with self.lock:
            self.ttft_sum += seconds
            for index, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    self.ttft_counts[index] += 1
            self.ttft_counts[-1] += 1

    def metrics_text(self) -> str:
        label = f'model_name="{self.model}"'
        with self.lock:
            lines = [
                "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.",
                "# TYPE vllm:kv_cache_usage_perc gauge",
                f"vllm:kv_cache_usage_perc{{{label}}} {min(1.0, self.running / self.kv_slots)}",
                "# HELP vllm:num_requests_running Number of requests currently running.",
                "# TYPE vllm:num_requests_running gauge",
                f"vllm:num_requests_running{{{label}}} {float(self.running)}",
                "# HELP vllm:num_requests_waiting Number of requests waiting to be scheduled.",
                "# TYPE vllm:num_requests_waiting gauge",
                f"vllm:num_requests_waiting{{{label}}} {float(self.waiting)}",
                "# HELP vllm:num_preemptions_total Cumulative number of preemptions.",
                "# TYPE vllm:num_preemptions_total counter",
                f"vllm:num_preemptions_total{{{label}}} {self.preemptions}",
                "# HELP vllm:prefix_cache_hits_total Cumulative prefix-cache hits.",
                "# TYPE vllm:prefix_cache_hits_total counter",
                f"vllm:prefix_cache_hits_total{{{label}}} 0",
                "# HELP vllm:prefix_cache_queries_total Cumulative prefix-cache lookups.",
                "# TYPE vllm:prefix_cache_queries_total counter",
                f"vllm:prefix_cache_queries_total{{{label}}} {self.prompt_tokens}",
                "# HELP vllm:prompt_tokens_total Cumulative logical prompt tokens.",
                "# TYPE vllm:prompt_tokens_total counter",
                f"vllm:prompt_tokens_total{{{label}}} {self.prompt_tokens}",
                "# HELP vllm:generation_tokens_total Cumulative computed generation tokens.",
                "# TYPE vllm:generation_tokens_total counter",
                f"vllm:generation_tokens_total{{{label}}} {self.generation_tokens}",
                *(
                    line
                    for name, total in (
                        ("queue", self.queue_seconds),
                        ("prefill", self.prefill_seconds),
                    )
                    for line in (
                        f"# TYPE vllm:request_{name}_time_seconds histogram",
                        f"vllm:request_{name}_time_seconds_sum{{{label}}} {total}",
                        f"vllm:request_{name}_time_seconds_count{{{label}}} {self.success}",
                    )
                ),
                "# TYPE vllm:cache_config_info gauge",
                CACHE_CONFIG_INFO,
                "# HELP vllm:request_success_total Cumulative successfully completed requests.",
                "# TYPE vllm:request_success_total counter",
                f"vllm:request_success_total{{{label}}} {self.success}",
                "# HELP vllm:time_to_first_token_seconds Time to first token in seconds.",
                "# TYPE vllm:time_to_first_token_seconds histogram",
            ]
            for bound, count in zip(BUCKETS, self.ttft_counts, strict=False):
                lines.append(
                    f'vllm:time_to_first_token_seconds_bucket{{{label},le="{bound}"}} {count}'
                )
            total = self.ttft_counts[-1]
            lines.append(f'vllm:time_to_first_token_seconds_bucket{{{label},le="+Inf"}} {total}')
            lines.append(f"vllm:time_to_first_token_seconds_count{{{label}}} {total}")
            lines.append(f"vllm:time_to_first_token_seconds_sum{{{label}}} {self.ttft_sum}")
        return "\n".join(lines) + "\n"


def make_handler(state: State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if state.api_key is None:
                return True
            return self.headers.get("Authorization") == f"Bearer {state.api_key}"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, b"", "text/plain")
            elif self.path == "/metrics":
                self._send(200, state.metrics_text().encode(), "text/plain; version=0.0.4")
            elif self.path == "/v1/models":
                if not self._authorized():
                    self._send(401, b'{"error":"unauthorized"}', "application/json")
                    return
                card = {"id": state.model, "object": "model", "max_model_len": state.max_model_len}
                body = json.dumps({"object": "list", "data": [card]}).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b"{}", "application/json")

        def _profile(self) -> None:
            if state.trace_dir is None:
                self._send(404, b"{}", "application/json")
                return
            if self.path == "/stop_profile":
                name = f"fake_{os.getpid()}.{int(time.time() * 1000)}.pt.trace.json.gz"
                (Path(state.trace_dir) / name).write_bytes(
                    gzip.compress(SYNTHETIC_TRACE.read_bytes())
                )
            self._send(200, b"", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path in ("/start_profile", "/stop_profile"):
                self._profile()
                return
            if self.path not in ("/v1/completions", "/v1/chat/completions", "/tokenize"):
                self._send(404, b"{}", "application/json")
                return
            if self.path != "/tokenize" and not self._authorized():
                self._send(401, b'{"error":"unauthorized"}', "application/json")
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            if not all(valid_image_url(url) for url in image_urls(request)):
                error = {"message": "image_url must be a base64 data URL", "code": 400}
                self._send(400, json.dumps({"error": error}).encode(), "application/json")
                return
            chat = "messages" in request
            prompt_tokens = (
                chat_words(request) if chat else len(str(request.get("prompt", "")).split())
            )
            if self.path == "/tokenize":
                body = {
                    "count": prompt_tokens,
                    "max_model_len": state.max_model_len,
                    "tokens": list(range(prompt_tokens)),
                }
                self._send(200, json.dumps(body).encode(), "application/json")
                return
            count = int(request.get("max_tokens", 16))
            total = prompt_tokens + count
            if total > state.max_model_len:
                message = (
                    f"This model's maximum context length is {state.max_model_len} tokens. "
                    f"However, you requested {total} tokens ({prompt_tokens} in the prompt, "
                    f"{count} in the completion). Please reduce the length of the prompt "
                    "or completion."
                )
                error = {"message": message, "type": "BadRequestError", "param": None, "code": 400}
                self._send(400, json.dumps({"error": error}).encode(), "application/json")
                return
            if request.get("tools") and not state.tools_enabled:
                message = (
                    '"auto" tool choice requires --enable-auto-tool-choice '
                    "and --tool-call-parser to be set"
                )
                error = {"message": message, "type": "BadRequestError", "param": None, "code": 400}
                self._send(400, json.dumps({"error": error}).encode(), "application/json")
                return
            if request.get("tools"):
                pieces = tool_call_pieces(request["tools"][0])
                finish, count = "tool_calls", len(pieces)
            else:
                key = "delta" if chat else "text"
                word = "alt" if state.drift and len(prompt_text(request)) % 2 else "tok"
                pieces = [
                    {key: {"content": f"{word}{i} "} if chat else f"{word}{i} "}
                    for i in range(count)
                ]
                finish = "length"
            started = time.monotonic()
            with state.lock:
                state.waiting += 1
            state.slots.acquire()
            admitted = time.monotonic()
            with state.lock:
                state.waiting -= 1
                state.running += 1
                if state.running > state.kv_slots:
                    state.preemptions += 1
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                base = {"id": "cmpl-fake", "object": "text_completion", "model": state.model}
                if chat:
                    role = {"index": 0, "delta": {"role": "assistant", "content": ""}}
                    self._frame({**base, "choices": [role]})
                state.prefill(prompt_tokens, prompt_text(request))
                prefilled = time.monotonic()
                due = time.monotonic()
                for index, piece in enumerate(pieces):
                    due += state.step_seconds()  # paced on a schedule, so overhead never adds up
                    time.sleep(max(0.0, due - time.monotonic()))
                    if index == 0:
                        state.observe_ttft(time.monotonic() - started)
                    last = index == count - 1
                    choice = {"index": 0, **piece, "finish_reason": finish if last else None}
                    self._frame({**base, "choices": [choice]})
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": count,
                    "total_tokens": prompt_tokens + count,
                }
                self._frame({**base, "choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                with state.lock:
                    state.success += 1
                    state.queue_seconds += admitted - started
                    state.prefill_seconds += prefilled - admitted
                    state.prompt_tokens += prompt_tokens
                    state.generation_tokens += count
            finally:
                with state.lock:
                    state.running -= 1
                state.slots.release()

        def _frame(self, payload: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


class BurstServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128  # the default backlog drops a burst and costs a 1s SYN retry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-auto-tool-choice", action="store_true")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--profiler-config", default="{}")
    arguments, _ignored = parser.parse_known_args()
    time.sleep(float(os.environ.get("FAKE_LOAD_SECONDS", "0")))  # loading the model
    if kernel_line := os.environ.get("FAKE_QUANT_KERNEL_LINE"):
        print(kernel_line, flush=True)  # what vLLM logs at load time
    if arguments.kv_cache_dtype == "fp8":
        sys.exit("ValueError: fp8 KV cache is not supported here")
    state = State(
        arguments.served_model_name,
        os.environ.get("VLLM_API_KEY"),
        arguments.max_num_seqs,
        arguments.gpu_memory_utilization * 10,
        arguments.max_model_len,
        arguments.max_num_batched_tokens,
        arguments.enable_prefix_caching,
        arguments.enable_auto_tool_choice and arguments.tool_call_parser is not None,
        json.loads(arguments.profiler_config).get("torch_profiler_dir"),
        arguments.kv_cache_dtype == "fp8_e4m3",
    )
    server = BurstServer((arguments.host, arguments.port), make_handler(state))
    server.serve_forever()


if __name__ == "__main__":
    main()
