"""A stdlib-only stand-in for ``llama-server``, used by the end-to-end llama.cpp tests.

It takes the real argument vector (unknown flags are ignored) and imports nothing from Tensward.
``--version`` (to stderr, as llama-server does) and ``--help`` print the texts of build 11669
stored beside this file; ``FAKE_LLAMA_HELP_DROP=--n-cpu-moe,--load-mode`` removes those options
from ``--help``, as an older build lacks them.

It serves ``/health`` (503 for ``FAKE_START_DELAY_S`` seconds), ``/v1/models``, streaming
``/v1/chat/completions`` and ``/v1/completions`` (the last event carries ``usage`` and
``timings`` and no choices), ``POST /tokenize`` and ``/apply-template`` (one token per word) and
``/metrics`` with llama-server's ``llamacpp:`` families. The API key (env ``LLAMA_API_KEY``)
protects the ``/v1`` endpoints and the POST endpoints.

The model file is read only for its GGUF header: with ``-c`` unset the context is the model's
training context, as with ``--fit off``. ``-np`` caps the requests served at once (the rest are
deferred), and ``-kvu`` / ``-no-kvu`` (split by default when ``-np`` is given, unified when not,
as at build 11669) decide whether a request may use the whole context or its slot's share.
Its start-up log has llama-server's own lines at ``-lv 4`` with the default timestamp and level
prefix (copied from real b11669 logs, paths removed). ``--fit on`` (the default) lets the fake cut
an unset ``-c`` to ``FAKE_FIT_CTX`` (to ``FAKE_MTP_FIT_CTX`` beside an MTP draft context, which
fails to start with ``--fit off`` and a longer ``-c``) and, with no ``-ngl``, offload only
``FAKE_FIT_LAYERS`` layers
(``FAKE_FIT_MOE`` of them with the experts on the CPU),
each logged as ``--fit`` logs it; ``FAKE_BACKEND=metal`` names the device buffers ``MTL0``;
``FAKE_FINGERPRINT=vllm|ollama|lmstudio`` makes the endpoints answer like that server
(``/props`` absent; Ollama also answers ``/api/version``, ``/api/tags``, ``/api/ps`` and
``POST /api/show`` for one model, loaded unless ``FAKE_OLLAMA_LOADED=0``, at
``FAKE_OLLAMA_CTX`` tokens, none with
``FAKE_OLLAMA_EMPTY=1``, two with ``FAKE_OLLAMA_TWO=1``, whose Modelfile names
``FAKE_OLLAMA_BLOB`` and whose parameters are ``FAKE_OLLAMA_PARAMS``), and ``FAKE_SEEN_LOG=path``
appends each request it served; ``FAKE_BACKEND=cpu`` is a CPU-only build, which still prints
``offloaded N/N layers to GPU``. ``FAKE_BACKEND=apple`` prints the Metal
device line and a graph line, and ``--list-devices`` lists a Metal device of
``FAKE_METAL_WS_MIB`` (18186) MiB, else none; ``FAKE_METAL_PRIVATE=1`` books the compute buffer as
``MTL0_Private``; ``FAKE_METAL_DEVICE=1`` makes a CPU-only build print the Metal device lines too,
as a Mac does under ``-dev none``. ``GET /props`` answers like llama-server's and
``GET /slots`` lists the slots with the tokens each busy one holds.
``FAKE_METAL_OOM=1`` makes every completion after the first ``FAKE_METAL_OOM_AFTER`` log Metal's
out-of-memory line and answer 500,
Tensward's tool check (``X-Tensward-Probe: tool-check``) is answered in text the first time, as a
model declining tools that do not fit, then with a call to the request's first tool;
``FAKE_NO_TOOL_CALLS=1`` answers every probe in text,
``FAKE_HANG_AFTER=N`` makes every completion after the first N never answer,
``FAKE_ABORT_AFTER=N`` makes the server abort as b12246 did under draft-mtp at -np 4 (L27),
printing the abort line and a backtrace, after the first N,
``FAKE_KV_FULL=1`` makes every fourth request log the KV-full retry line and still serve it,
and refuses every fourth other one with
``400 exceed_context_size_error``; ``FAKE_THINKS=1`` streams
``reasoning_content`` instead of content until ``max_tokens``, unless the request sets
``chat_template_kwargs.enable_thinking`` to false.

A decode step costs ``TOKEN_DELAY_S * (1 + running / CONTENTION_SEQS)``, paced on a running
deadline so the time spent sending a frame is not added to the step. With ``FAKE_BW_HOST_GBS``
(and ``FAKE_BW_DEV_GBS``) it costs the bytes a token reads from device memory over the device
bandwidth plus those from host memory over the host's, the bytes taken from the GGUF's tensor
sizes, ``-ngl`` and ``-ncmoe`` (the model's last ``-ngl`` layers sit on the device, the first
``-ncmoe`` layers keep their experts on the host, the token embedding stays on the host).
With ``FAKE_PCIE_GBS`` a prefill also pays, per ubatch (``-ub``), the time the host weights take
to cross a link of that speed; ``FAKE_PREFILL_S_PER_PASS`` adds that many seconds per ubatch.
``FAKE_LAUNCH_LOG=path`` appends a line per launch with its ``GLIBC_TUNABLES``.
``FAKE_REWORD_FLAG=--flag`` words every other answer differently when the launch has that
flag; ``FAKE_FAST_FLAG=--flag`` makes its decode steps take 0.6 of the time;
``FAKE_FEWER_CALLS_FLAG=--flag`` answers in text one in ``FAKE_FEWER_CALLS_EVERY`` (default 3)
of the prompts that call a tool. ``FAKE_ECHO=1``
answers with the prompt's words in order, as an answer that copies its input does.

``FAKE_SNAPSHOTS=N`` makes the server a hybrid model's: a request that offers tools counts N more
tokens (a shared start of system prompt and tools), and the log has the slot lines at ``-lv 4``
(``new prompt``, ``restored context checkpoint``) with the spacing line. The first four slots
to fill (seven with ``-b 2048`` or more) hold a snapshot at N; a slot filling later holds none
there and resumes below it on its later requests, unless ``-cms 0`` is given, which gives
every slot the snapshot.

``FAKE_SIMILARITY=1`` adds the slot-pick lines: each request after a slot's first is picked "by LCP
similarity" and read whole again, unless the launch has ``-sps 0``, which picks "by LRU".

``FAKE_REASONING=1`` gives each prompt its own greedy reasoning (with ``FAKE_THINKS=1``): token
``i`` is ``r<seed>x<i>``, the seed taken from the prompt's first three words not ending in a
colon, and its near tie ``a<seed>x<i>``. ``/apply-template`` then renders only the messages'
text, as the usage counts it, ``/tokenize`` returns ids that decode (and pieces with
``with_pieces``), and ``POST /completion`` answers a token-id prompt with ``n_probs`` top log
probabilities of the next token by that rule. ``FAKE_FLIP_FLAG=--flag`` with
``FAKE_FLIP_EVERY=N`` writes the near tie at every N-th token (counted from the seed) when the
launch has that flag; ``FAKE_WRONG_FLAG=--flag`` writes another prompt's reasoning.
``FAKE_GREEDY=1`` writes final answers by the same rule (``/props`` then names ``<eot>`` as the
end of turn); a prompt with the word ``FAKE_TOOL_WORD`` in it is answered with one ``lookup``
call whose ``q`` holds TOOL_WORDS such tokens, and ``/apply-template`` renders a tool call as
those tokens and ``<eot>``. ``FAKE_FAR_FLAG=--flag`` writes, at every ``FAKE_FLIP_EVERY``-th
token, one the model rates 1 in 40 against its pick; ``FAKE_FLIP_ONLY=word`` limits the flips
to prompts with that word; ``FAKE_NO_PROBS=1`` answers ``/completion`` with no probabilities.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINGERPRINT = os.environ.get("FAKE_FINGERPRINT", "llamacpp")
SEEN_LOG = os.environ.get("FAKE_SEEN_LOG")
OWNERS = {
    "llamacpp": "llamacpp", "vllm": "vllm", "ollama": "library", "lmstudio": "organization_owner",
}  # fmt: skip
OLLAMA_MODEL = {
    "name": "fake:1b", "model": "fake:1b", "size": 2**30,
    "details": {"format": "gguf", "quantization_level": "Q8_0"},
}  # fmt: skip
OLLAMA_LOADED = {
    **OLLAMA_MODEL, "size_vram": 2**30, "context_length": 4096,
    "expires_at": "2999-01-01T00:00:00Z",
}  # fmt: skip
TOKEN_DELAY_S = float(os.environ.get("FAKE_TOKEN_DELAY_S", "0.0007"))
CONTENTION_SEQS = 2.0
PREFILL_S_PER_TOKEN = 0.00002
PREFILL_S_PER_PASS = float(os.environ.get("FAKE_PREFILL_S_PER_PASS", "0"))
PREFIX_WORDS = 100
CACHED_PREFILL_TOKENS = 16
DEFAULT_PARALLEL = 4
DEFAULT_CONTEXT = 4096
GREEDY = os.environ.get("FAKE_GREEDY") == "1"
REASONING = os.environ.get("FAKE_REASONING") == "1" or GREEDY
REASONING_WORD = re.compile(r"^[raf]\d+x\d+$")
TOOL_WORD = os.environ.get("FAKE_TOOL_WORD")
TOOL_WORDS = 4
END_OF_TURN = "<eot>"
BUILD_NUMBER = os.environ.get("FAKE_LLAMA_BUILD", "11669")
BUILD = f"{BUILD_NUMBER} (620d2225c)"
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}

OOM_LINE = "error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
LOG_TEMPLATES = {
    "build": "cmn  common_param: common_params_print_info: build " + BUILD + " with GNU "
    "13.3.0 for Linux x86_64",
    # source = "b11982 M4 capture"
    "apple_build": "cmn  common_param: common_params_print_info: build 11982 (8a199ad31) with "
    "AppleClang 21.0.0.21000101 for Darwin arm64",
    "apple_device": "cmn  common_param:   - MTL0    : Apple M4 ({mib} MiB, {free} MiB free)",
    "apple_system": "cmn  common_param: system_info: n_threads = 4 (n_threads_batch = 4) / 10 | "
    "MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | "
    "ACCELERATE = 1 | REPACK = 1 |",
    "system": {
        "gpu": "cmn  common_param: system_info: n_threads = 6 (n_threads_batch = 6) / 12 | CUDA : "
        "ARCHS = 500,610,700,750,800,860,890,900,1200 | USE_GRAPHS = 1 | FA_QUANTS = "
        "q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1",
        "cpu": "cmn  common_param: system_info: n_threads = 6 (n_threads_batch = 6) / 8 | CPU : "
        "SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | LLAMAFILE = "
        "1 | OPENMP = 1 | REPACK = 1 |",
    },
    "loading": "srv    load_model: loading model '{name}'",
    "fit_start": "cmn  common_init_: fitting params to device memory ...",
    "fit_done": "common_fit_params: successfully fit params to free device memory",
    # common/fit.cpp:434 and :672, :709: logged at the same level as the lines above
    "fit_context": "common_params_fit_impl: context size reduced from {before} to {after} -> need "
    "{mib} MiB less memory in total",
    "fit_layers": "common_params_fit_impl: filling {kind} layers back-to-front:",
    "fit_set": "common_params_fit_impl: set ngl_per_device[0].n_layer={layers}",
    "fit_summary": "common_params_fit_impl:   - CUDA0 (NVIDIA L4): {layers:>2} layers{overflow}, "
    "{used:>6} MiB used, {free:>6} MiB free",
    "file_size": "print_info: file size   = {gib:.2f} GiB (6.56 BPW) ",
    "load_mode": "load_tensors: loading model tensors, this can take a while... (load_mode = mmap)",
    "offloaded": "load_tensors: offloaded {n}/{m} layers to GPU",
    "model_buffer": "load_tensors: {device:>12} model buffer size = {mib:>8.2f} MiB",
    "context": [
        "llama_context: constructing llama_context",
        "llama_context: n_seq_max             = {parallel}",
        "llama_context: n_ctx                 = {pool}",
        "llama_context: n_ctx_seq             = {slot}",
        "llama_context: n_batch               = 2048",
        "llama_context: n_ubatch              = {ubatch}",
        "llama_context: flash_attn            = {flash}",
        "llama_context: kv_unified            = {unified}",
    ],
    "output_buffer": "llama_context: {device:>10}  output buffer size = {mib:>8.2f} MiB",
    "kv_buffer": "llama_kv_cache: {device:>10} KV buffer size = {mib:>8.2f} MiB",
    "kv_size": "llama_kv_cache: size = {mib:.2f} MiB ({cells:>6} cells, {layers:>3} layers, "
    "{seqs}), K ({kind}): {half:>7.2f} MiB, V ({kind}): {half:>7.2f} MiB",
    "flash": "resolve_fused_ops: Flash Attention enabled",
    "compute_buffer": "sched_reserve: {device:>10} compute buffer size = {mib:>8.2f} MiB",
    "slots": "srv    load_model: initializing, n_slots = {parallel}, n_ctx_slot = {slot}, "
    "kv_unified = '{unified}'",
    # llama-context.cpp:873, printed once pp and tg graphs agree
    "graph": "sched_reserve: graph: nodes = 1534, splits = 2, input objects = 3, input tensors = 4",
    "kv_full": "srv  update_slots: failed to find free space in the KV cache, retrying with "
    "smaller batch size, off = 0, n_batch = 2048, ret = 1",
    "loaded": "srv  llama_server: model loaded",
    "listening": "srv  llama_server: listening on http://{host}:{port}",
}
MIB = 2**20


def read_gguf_meta(path: str) -> dict | None:
    """The model's context length, layer count, K width per layer and expert counts
    (``context``, ``layers``, ``kv_width``, ``experts``, ``experts_used``) and its tensors'
    sizes (``tensors``: name and bytes, from the offsets), read from the GGUF header; None when
    the file is not a GGUF or has no ``<arch>.context_length``."""
    try:
        with open(path, "rb") as file:
            if file.read(4) != b"GGUF":
                return None
            _version, tensors, count = struct.unpack("<IQQ", file.read(20))

            def scalar(kind: int) -> object:
                code = SCALARS[kind]
                return struct.unpack("<" + code, file.read(struct.calcsize(code)))[0]

            def value(kind: int) -> object:
                if kind in SCALARS:
                    return scalar(kind)
                if kind == 8:
                    (size,) = struct.unpack("<Q", file.read(8))
                    return file.read(size).decode("utf-8", "replace")
                inner, items = struct.unpack("<IQ", file.read(12))
                return [value(inner) for _ in range(items)]

            found: dict[str, object] = {}
            for _ in range(count):
                (size,) = struct.unpack("<Q", file.read(8))
                key = file.read(size).decode()
                (kind,) = struct.unpack("<I", file.read(4))
                found[key] = value(kind)
            offsets = []
            for _ in range(tensors):
                (size,) = struct.unpack("<Q", file.read(8))
                name = file.read(size).decode()
                (dims,) = struct.unpack("<I", file.read(4))
                file.seek(8 * dims + 4, 1)
                (offset,) = struct.unpack("<Q", file.read(8))
                offsets.append((name, offset))
            data = -(-file.tell() // 32) * 32
            ends = [offset for _, offset in offsets[1:]] + [os.path.getsize(path) - data]
            sizes = [(name, end - offset) for (name, offset), end in zip(offsets, ends)]
            arch = found.get("general.architecture")
            if arch is None or f"{arch}.context_length" not in found:
                return None
            declared = found.get(f"{arch}.attention.head_count_kv", 8)
            heads = max(declared) if isinstance(declared, list) else int(declared)  # type: ignore[call-overload]
            width = int(found.get(f"{arch}.attention.key_length", 128))  # type: ignore[call-overload]
            return {
                "context": int(found[f"{arch}.context_length"]),  # type: ignore[call-overload]
                "layers": int(found.get(f"{arch}.block_count", 28)),  # type: ignore[call-overload]
                "kv_width": heads * width,
                "experts": int(found.get(f"{arch}.expert_count", 0)),  # type: ignore[call-overload]
                "experts_used": int(found.get(f"{arch}.expert_used_count", 0)),  # type: ignore[call-overload]
                "tensors": sizes,
            }
    except (OSError, struct.error, KeyError, ValueError):
        return None


def help_text(drop: set[str]) -> str:
    kept: list[str] = []
    dropping = False
    for line in (HERE / "llamacpp_b11669_help.txt").read_text().splitlines():
        if line.startswith("-") and not line.startswith("----- "):
            spec = re.split(r"\s{2,}(?=[^-\s])", line, maxsplit=1)[0]
            names = {piece.split()[0] for piece in spec.split(",") if piece.split()}
            dropping = bool(names & drop)
        elif line.startswith("-----"):
            dropping = False
        if not dropping:
            kept.append(line)
    return "\n".join(kept) + "\n"


def message_text(message: dict) -> str:
    if GREEDY and message.get("tool_calls"):
        calls = [json.loads(call["function"]["arguments"])["q"] for call in message["tool_calls"]]
        return " ".join([*calls, END_OF_TURN])
    content = message.get("content", "")
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if part.get("type") == "text")
    return str(content)


def seed_of(words: list[str]) -> int:
    return sum(map(ord, " ".join([w for w in words if not w.endswith(":")][:3]))) % 1000


def prompt_text(request: dict) -> str:
    if "messages" in request:
        return " ".join(message_text(message) for message in request["messages"])
    return str(request.get("prompt", ""))


class Slots:
    """A first-come first-served counting semaphore."""

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


class State:
    def __init__(
        self, alias: str, parallel: int, context: int, unified: bool, cache: bool, model_path: str
    ) -> None:
        self.alias = alias
        self.model_path = model_path
        self.parallel = parallel
        self.context = context
        self.unified = unified
        self.cache = cache
        self.api_key = os.environ.get("LLAMA_API_KEY")
        self.ready_at = time.monotonic() + float(os.environ.get("FAKE_START_DELAY_S", "0"))
        self.slots = Slots(parallel)
        self.lock = threading.Lock()
        self.running = 0
        self.deferred = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.generated = 0
        self.decodes = 0.0
        self.busy = 0.0
        self.seen_prefixes: set[tuple[str, ...]] = set()
        self.held: dict[int, int] = {}
        self.requests = 0
        self.probes = 0
        self.log: Log | None = None
        self.paced_step_s: float | None = None
        self.link_s = 0.0
        self.ubatch = 512
        self.reworded = False
        self.fewer_calls = 0
        self.speed = 1.0
        self.flip_every = 0
        self.far = False
        self.wrong = False
        self.snapshots = int(os.environ.get("FAKE_SNAPSHOTS", "0"))
        self.spacing = 8192
        self.fits = 4
        self.similarity = os.environ.get("FAKE_SIMILARITY") == "1"
        self.picks_by_similarity = True
        self.served: set[int] = set()
        self.idle_slots = list(range(parallel))
        self.fill: dict[int, bool] = {}
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def take_slot(self) -> int:
        with self.lock:
            self.idle_slots.sort()
            return self.idle_slots.pop(0)

    def give_slot(self, slot: int) -> None:
        with self.lock:
            self.idle_slots.append(slot)

    def note_pick(self, slot: int, tokens: int) -> None:
        """The slot-pick lines of one request, and the full read that follows a similarity pick."""
        if self.log is None:
            return
        with self.lock:
            again = slot in self.served
            self.served.add(slot)
            task = self.requests * 2
        head = f"slot       select: id {slot:>2} | task -1 | "
        if not again or not self.picks_by_similarity:
            self.log.line(f"{head}selected slot by LRU, n_tokens = 0, t_last = -1")
            return
        self.log.line(f"{head} - checking sim = 0.675 ({tokens // 2}/{tokens}) > 0.100")
        self.log.line(
            f"{head}selected slot by LCP similarity, f_sim_best = 0.675 (> 0.100 thold), "
            "f_keep = 0.663"
        )
        self.log.line(
            f"slot   operator(): id {slot:>2} | task {task} | forcing full prompt re-processing "
            "due to lack of cache data (likely due to SWA or hybrid/recurrent memory)"
        )

    def note_prompt(self, slot: int, tokens: int, tools: bool) -> None:
        """The slot lines of one request: the prompt, and where the slot resumes it."""
        if self.log is None:
            return
        with self.lock:
            task = self.requests * 2
            seen = slot in self.fill
            if not seen:
                self.fill[slot] = len(self.fill) < self.fits or self.spacing == 0
            holds = self.fill[slot]
        head = f"slot   operator(): id {slot:>2} | task {task} | "
        self.log.line(
            f"{head}new prompt, n_ctx_slot = {self.slot_context}, n_keep = 0, "
            f"task.n_tokens = {tokens}"
        )
        if tools and seen:
            at = self.snapshots if holds else 100 + 37 * slot
            self.log.line(
                f"{head}restored context checkpoint (pos_min = {at - 1}, pos_max = {at - 1}, "
                f"n_tokens = {at}, n_past = {at}, size = 149.626 MiB)"
            )

    def ids(self, words: list[str]) -> list[int]:
        with self.lock:
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)
            return [self.vocab[word] for word in words]

    def reasoning_word(self, seed: int, index: int, flips: bool = True) -> str:
        seed += self.wrong
        near = flips and self.flip_every and (index + seed) % self.flip_every == 0
        return f"{('f' if self.far else 'a') if near else 'r'}{seed}x{index}"

    def props(self) -> dict:
        return {
            "default_generation_settings": {"n_ctx": self.slot_context},
            "total_slots": self.parallel,
            "model_alias": self.alias,
            "model_path": self.model_path,
            "modalities": {"vision": False, "video": False, "audio": False},
            "chat_template_caps": {"supports_tools": True, "supports_system_role": True},
            "build_info": os.environ.get("FAKE_BUILD", "b11669-620d2225c"),
            **({"eos_token": END_OF_TURN} if GREEDY else {}),
        }

    @property
    def slot_context(self) -> int:
        return self.context if self.unified else self.context // self.parallel

    def step_seconds(self) -> float:
        if self.paced_step_s is not None:
            return self.paced_step_s
        return TOKEN_DELAY_S * self.speed * (1 + self.running / CONTENTION_SEQS)

    def prefill(self, tokens: int, text: str) -> int:
        """Prefill the prompt; returns the tokens served from the prompt cache."""
        remaining, cached = tokens, 0
        if self.cache:
            prefix = tuple(text.split()[:PREFIX_WORDS])
            with self.lock:
                hit = prefix in self.seen_prefixes
                self.seen_prefixes.add(prefix)
            if hit:
                cached = max(tokens - CACHED_PREFILL_TOKENS, 0)
                remaining = tokens - cached
        passes = -(-remaining // self.ubatch)
        time.sleep(remaining * PREFILL_S_PER_TOKEN + passes * (self.link_s + PREFILL_S_PER_PASS))
        return cached

    def slots_json(self) -> list[dict]:
        with self.lock:
            held = sorted(self.held.values(), reverse=True)
        return [
            {
                "id": index,
                "n_ctx": self.slot_context,
                "is_processing": index < len(held),
                "n_prompt_tokens": held[index] if index < len(held) else 0,
            }
            for index in range(self.parallel)
        ]

    def metrics_text(self) -> str:
        with self.lock:
            families = (
                ("counter", "prompt_tokens_total", self.prompt_tokens),
                ("counter", "prompt_tokens_cached_total", self.cached_tokens),
                ("counter", "tokens_predicted_total", self.generated),
                ("counter", "n_decode_total", self.decodes),
                ("gauge", "requests_processing", self.running),
                ("gauge", "requests_deferred", self.deferred),
                ("gauge", "n_busy_slots_per_decode", self.busy / max(self.decodes, 1.0)),
            )
        return "".join(
            f"# HELP llamacpp:{name} {name}\n# TYPE llamacpp:{name} {kind}\n"
            f"llamacpp:{name} {value}\n"
            for kind, name, value in families
        )


def make_handler(state: State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

        def _send(self, status: int, payload: object) -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, kind: str, message: str) -> None:
            self._send(status, {"error": {"code": status, "message": message, "type": kind}})

        def _authorized(self) -> bool:
            if state.api_key is None:
                return True
            if self.headers.get("Authorization") == f"Bearer {state.api_key}":
                return True
            self._error(401, "authentication_error", "Invalid API Key")
            return False

        def do_GET(self) -> None:  # noqa: N802
            if SEEN_LOG:
                with open(SEEN_LOG, "a", encoding="utf-8") as seen:
                    seen.write(f"GET {self.path}\n")
            if self.path == "/api/version" and FINGERPRINT == "ollama":
                self._send(200, {"version": "0.5.1"})
            elif self.path == "/api/tags" and FINGERPRINT == "ollama":
                empty = os.environ.get("FAKE_OLLAMA_EMPTY") == "1"
                two = os.environ.get("FAKE_OLLAMA_TWO") == "1"
                models = (
                    [OLLAMA_MODEL, {**OLLAMA_MODEL, "name": "fake:2b"}] if two else [OLLAMA_MODEL]
                )
                self._send(200, {"models": [] if empty else models})
            elif self.path == "/api/ps" and FINGERPRINT == "ollama":
                loaded = os.environ.get("FAKE_OLLAMA_LOADED", "1") == "1"
                context = int(os.environ.get("FAKE_OLLAMA_CTX", "4096"))
                self._send(
                    200,
                    {"models": [{**OLLAMA_LOADED, "context_length": context}] if loaded else []},
                )
            elif self.path == "/props" and FINGERPRINT != "llamacpp":
                self._error(404, "not_found_error", "File Not Found")
            elif self.path == "/health":
                if time.monotonic() < state.ready_at:
                    self._error(503, "unavailable_error", "Loading model")
                else:
                    self._send(200, {"status": "ok"})
            elif self.path == "/metrics":
                body = state.metrics_text().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/slots":
                self._send(200, state.slots_json())
            elif self.path == "/props":
                if self._authorized():
                    self._send(200, state.props())
            elif self.path == "/v1/models":
                if not self._authorized():
                    return
                card = {"id": state.alias, "object": "model", "owned_by": OWNERS[FINGERPRINT]}
                models = [{"model": state.alias, "details": {"format": "gguf"}}]
                self._send(200, {"object": "list", "data": [card], "models": models})
            else:
                self._error(404, "not_found_error", "File Not Found")

        def do_POST(self) -> None:  # noqa: N802
            if SEEN_LOG:
                with open(SEEN_LOG, "a", encoding="utf-8") as seen:
                    seen.write(f"POST {self.path}\n")
            if self.path == "/api/show" and FINGERPRINT == "ollama":
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                blob = os.environ.get("FAKE_OLLAMA_BLOB", "/x/blobs/sha256-" + "0" * 64)
                self._send(200, {
                    "modelfile": f"# Modelfile generated by ollama\nFROM {blob}\n",
                    "parameters": os.environ.get("FAKE_OLLAMA_PARAMS", ""),
                    "template_source": os.environ.get("FAKE_OLLAMA_TEMPLATE", "go_template"),
                    "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960},
                })  # fmt: skip
                return
            known = (
                "/v1/chat/completions", "/v1/completions", "/tokenize", "/apply-template",
                *(("/completion",) if REASONING else ()),
            )  # fmt: skip
            if self.path not in known:
                self._error(404, "not_found_error", "File Not Found")
                return
            if not self._authorized():
                return
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/tokenize" and REASONING:
                pieces = re.findall(r"\S+\s*", str(request.get("content", "")))
                ids = state.ids([piece.strip() for piece in pieces])
                if request.get("with_pieces"):
                    self._send(200, {"tokens": [
                        {"id": i, "piece": piece} for i, piece in zip(ids, pieces)
                    ]})  # fmt: skip
                else:
                    self._send(200, {"tokens": ids})
            elif self.path == "/tokenize":
                tokens = len(str(request.get("content", "")).split())
                self._send(200, {"tokens": list(range(tokens))})
            elif self.path == "/apply-template" and REASONING:
                self._send(200, {"prompt": prompt_text(request)})
            elif self.path == "/completion":
                self._probe(request)
            elif self.path == "/apply-template":
                lines = [f"{m['role']}: {message_text(m)}" for m in request["messages"]]
                lines += [json.dumps(tool) for tool in request.get("tools", [])]
                self._send(200, {"prompt": "\n".join(lines)})
            else:
                self._complete(request, "messages" in request)

        def _probe(self, request: dict) -> None:
            words = [state.words[i] for i in request["prompt"]]
            reasoned = 0
            while reasoned < len(words) and REASONING_WORD.match(words[-1 - reasoned]):
                reasoned += 1
            seed = seed_of(words[: len(words) - reasoned])
            written = f"r{seed}x{reasoned}"
            if TOOL_WORD in words and reasoned == TOOL_WORDS:
                written = END_OF_TURN
            top = [(written, math.log(0.6)), (f"a{seed}x{reasoned}", math.log(0.35)),
                   (f"f{seed}x{reasoned}", math.log(0.015))]  # fmt: skip
            ids = state.ids([word for word, _ in top])
            listed = [
                {"id": i, "token": word, "bytes": list(word.encode()), "logprob": lp}
                for i, (word, lp) in zip(ids, top)
            ][: int(request.get("n_probs", 0))]
            found = {"content": top[0][0], "tokens_cached": len(words) - 1}
            if os.environ.get("FAKE_NO_PROBS") != "1":
                found["completion_probabilities"] = [{**listed[0], "top_logprobs": listed}]
            self._send(200, found)

        def _complete(self, request: dict, chat: bool) -> None:
            if self._tool_check(request, chat):
                return
            text = prompt_text(request)
            tools = bool(request.get("tools"))
            prompt_tokens = len(text.split()) + (state.snapshots if tools else 0)
            count = int(request.get("max_tokens", 16))
            with state.lock:
                state.requests += 1
                kv_full = os.environ.get("FAKE_KV_FULL") == "1"
                retried = kv_full and state.requests % 4 == 0
                refused = kv_full and state.requests % 4 == 2
            abort_after = os.environ.get("FAKE_ABORT_AFTER")
            if abort_after is not None and state.requests > int(abort_after):
                print(ABORT_LINE, file=sys.stderr, flush=True)
                print(
                    "/opt/llama/libggml-base.so.0(ggml_abort+0x15b)[0x77c544d170eb]",
                    file=sys.stderr, flush=True,
                )  # fmt: skip
                os._exit(134)
            hang_after = os.environ.get("FAKE_HANG_AFTER")
            if hang_after is not None and state.requests > int(hang_after):
                while True:
                    time.sleep(3600)
            if retried and state.log is not None:
                state.log.line(LOG_TEMPLATES["kv_full"])
            if os.environ.get("FAKE_METAL_OOM") == "1" and state.requests > int(
                os.environ.get("FAKE_METAL_OOM_AFTER", "0")
            ):
                if state.log is not None:
                    state.log.line(OOM_LINE)
                self._error(500, "server_error", "Compute error.")
                return
            if refused or prompt_tokens + count > state.slot_context:
                self._error(
                    400, "exceed_context_size_error",
                    "the request exceeds the available context size, try increasing it",
                )  # fmt: skip
                return
            with state.lock:
                state.deferred += 1
            state.slots.acquire()
            slot = state.take_slot()
            if state.similarity:
                state.note_pick(slot, prompt_tokens)
            if state.snapshots:
                state.note_prompt(slot, prompt_tokens, tools)
            started = time.monotonic()
            ticket = id(request)
            with state.lock:
                state.deferred -= 1
                state.running += 1
                state.held[ticket] = prompt_tokens
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                base = {
                    "id": "chatcmpl-fake",
                    "model": state.alias,
                    "object": "chat.completion.chunk",
                }
                if chat:
                    role = {"index": 0, "delta": {"role": "assistant", "content": None}}
                    self._frame({**base, "choices": [role]})
                kwargs = request.get("chat_template_kwargs") or {}
                thinks = (
                    chat
                    and os.environ.get("FAKE_THINKS") == "1"
                    and kwargs.get("enable_thinking") is not False
                )
                cached = state.prefill(prompt_tokens, text)
                only = os.environ.get("FAKE_FLIP_ONLY")
                flips = only is None or only in text.split()
                echo = text.split() if os.environ.get("FAKE_ECHO") == "1" else []
                prefilled = time.monotonic()
                due = prefilled
                calls = chat and GREEDY and TOOL_WORD in text.split()
                if calls and state.fewer_calls:
                    calls = seed_of(text.split()) % state.fewer_calls != 0
                if calls:
                    words = [
                        state.reasoning_word(seed_of(text.split()), i, flips)
                        for i in range(TOOL_WORDS)
                    ]
                    arguments = json.dumps({"q": " ".join(words)})
                    call = {"index": 0, "id": "call_0", "type": "function",
                            "function": {"name": "lookup", "arguments": arguments}}  # fmt: skip
                    choice = {"index": 0, "delta": {"tool_calls": [call]},
                              "finish_reason": "tool_calls"}  # fmt: skip
                    self._frame({**base, "choices": [choice]})
                for index in range(0 if calls else count):
                    due += state.step_seconds()
                    time.sleep(max(0.0, due - time.monotonic()))
                    with state.lock:
                        state.decodes += 1.0 / max(state.running, 1)
                        state.busy += 1.0 / max(state.running, 1) * state.running
                    word = (
                        f"{echo[index % len(echo)]} "
                        if echo
                        else f"{state.reasoning_word(seed_of(text.split()), index, flips)} "
                        if REASONING and (thinks or GREEDY)
                        else f"{'alt' if state.reworded and len(text) % 2 else 'tok'}{index} "
                    )
                    with state.lock:
                        state.held[ticket] = prompt_tokens + index + 1
                    if thinks:
                        delta = {"delta": {"reasoning_content": word}}
                    else:
                        delta = {"delta": {"content": word}} if chat else {"text": word}
                    last = index == count - 1
                    choice = {"index": 0, **delta, "finish_reason": "length" if last else None}
                    self._frame({**base, "choices": [choice]})
                finished = time.monotonic()
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": count,
                    "total_tokens": prompt_tokens + count,
                    "prompt_tokens_details": {"cached_tokens": cached},
                }
                timings = {
                    "cache_n": cached,
                    "prompt_n": prompt_tokens - cached,
                    "prompt_ms": (prefilled - started) * 1000,
                    "predicted_n": count,
                    "predicted_ms": (finished - prefilled) * 1000,
                }
                self._frame({**base, "choices": [], "usage": usage, "timings": timings})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                with state.lock:
                    state.prompt_tokens += prompt_tokens - cached
                    state.cached_tokens += cached
                    state.generated += count
            finally:
                with state.lock:
                    state.running -= 1
                    state.held.pop(ticket, None)
                state.give_slot(slot)
                state.slots.release()

        def _tool_check(self, request: dict, chat: bool) -> bool:
            """Answer Tensward's tool check (a request marked as its probe), the first one in text
            and the rest with a call to the request's first tool, touching none of the counters
            the workload is measured by."""
            asked = (
                chat and request.get("tools")
                and self.headers.get("X-Tensward-Probe") == "tool-check"
            )  # fmt: skip
            if not asked:
                return False
            with state.lock:
                state.probes += 1
                in_text = state.probes == 1 or os.environ.get("FAKE_NO_TOOL_CALLS") == "1"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            base = {"id": "chatcmpl-fake", "model": state.alias, "object": "chat.completion.chunk"}
            if in_text:
                said = "None of the available tools fit this request."
                choice = {"index": 0, "delta": {"content": said}, "finish_reason": "stop"}
            else:
                name = request["tools"][0]["function"]["name"]
                call = {"index": 0, "id": "call_0", "type": "function",
                        "function": {"name": name, "arguments": "{}"}}  # fmt: skip
                delta = {"tool_calls": [call]}
                choice = {"index": 0, "delta": delta, "finish_reason": "tool_calls"}
            self._frame({**base, "choices": [choice]})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return True

        def _frame(self, payload: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()

    return Handler


def exit_with_parent() -> None:
    """Exit once the process that started the server is gone: it runs in its own session, so a
    test run killed outright would otherwise leave it serving."""
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(0.5)
    os._exit(0)


class BurstServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def option(argv: list[str], *names: str) -> str | None:
    for index, word in enumerate(argv[:-1]):
        if word in names:
            return argv[index + 1]
    return None


ABORT_LINE = (
    "/__w/llama.cpp/llama.cpp/common/common.cpp:1634: failed to remove sequence 1 with "
    "p0=4030, p1=-1"
)


class Log:
    """The server log: llama-server's ``M.SS.mmm.uuu I `` prefix counts from the first line."""

    def __init__(self) -> None:
        self.began = time.monotonic()

    def line(self, text: str) -> None:
        elapsed = time.monotonic() - self.began
        minutes, seconds = divmod(int(elapsed), 60)
        micros = int((elapsed % 1) * 1e6)
        print(
            f"{minutes}.{seconds:02d}.{micros // 1000:03d}.{micros % 1000:03d} I {text}", flush=True
        )


def place_tensors(
    meta: dict, offloaded: int, cpu: bool, expert_layers: int
) -> tuple[float, float, float, float]:
    """(device MiB, host MiB, device MiB read per token, host MiB read per token) of the model
    with its last ``offloaded`` of ``layers + 1`` layers on the device (the output layer first)
    and the experts of its first ``expert_layers`` layers on the host. The token embedding
    always stays on the host and a token reads one row of it, nothing here."""
    total = meta["layers"] + 1
    host_layers = total - offloaded
    active = meta["experts_used"] / meta["experts"] if meta["experts"] else 1.0
    device = host = device_read = host_read = 0.0
    for name, size in meta["tensors"]:
        block = re.match(r"blk\.(\d+)\.", name)
        expert = "_exps" in name
        if name == "token_embd.weight":
            on_host, read = True, 0.0
        elif name.startswith("per_layer_token_embd."):
            on_host, read = cpu or offloaded < total, 0.0
        else:
            index = int(block[1]) if block else None
            on_host = cpu or (index is None and offloaded == 0)
            on_host = on_host or (index is not None and index < host_layers)
            on_host = on_host or (expert and index is not None and index < expert_layers)
            read = size * (active if expert else 1.0)
        if on_host:
            host, host_read = host + size, host_read + read
        else:
            device, device_read = device + size, device_read + read
    return device / MIB, host / MIB, device_read / MIB, host_read / MIB


def embedding_mib(meta: dict) -> float:
    return sum(size for name, size in meta["tensors"] if name == "token_embd.weight") / MIB


def startup_log(
    log: Log, argv: list[str], meta: dict, model: str
) -> tuple[int, bool, float | None, float, int]:
    """Print the start-up log and return the pool of ``-c`` (a cut one with ``--fit on``),
    whether the KV cache is unified, the seconds one decode step takes when
    ``FAKE_BW_HOST_GBS`` paces it by the bytes each memory serves, the seconds the host weights
    take to cross a link of ``FAKE_PCIE_GBS`` (paid once per ubatch of a prefill) and the
    ubatch size."""
    templates = LOG_TEMPLATES
    cpu = os.environ.get("FAKE_BACKEND") == "cpu"
    apple = os.environ.get("FAKE_BACKEND") == "apple"
    gpu = "MTL0" if apple or os.environ.get("FAKE_BACKEND") == "metal" else "CUDA0"
    layers = meta["layers"]
    total = layers + 1
    parallel_text = option(argv, "-np", "--parallel")
    parallel = int(parallel_text) if parallel_text and int(parallel_text) > 0 else DEFAULT_PARALLEL
    unified = parallel_text is None or "-kvu" in argv or "--kv-unified" in argv
    context_text = option(argv, "-c", "--ctx-size")
    pool = int(context_text) if context_text and int(context_text) > 0 else meta["context"]
    fit = option(argv, "--fit") != "off"
    expert_layers = int(option(argv, "--n-cpu-moe", "-ncmoe") or 0)
    ubatch = int(option(argv, "-ub", "--ubatch-size") or 512)
    if apple:
        working_set = int(os.environ.get("FAKE_METAL_WS_MIB", "18186"))
        log.line(templates["apple_build"])
        log.line(templates["apple_device"].format(mib=working_set, free=working_set - 1))
        log.line(templates["apple_system"])
    else:
        log.line(templates["build"])
        if cpu and os.environ.get("FAKE_METAL_DEVICE"):
            log.line(templates["apple_device"].format(mib=18186, free=18185))
        log.line(templates["system"]["cpu" if cpu else "gpu"])
    log.line(templates["loading"].format(name=Path(model).name))
    log.line(f"print_info: n_layer               = {meta['layers']}")
    ngl_text = option(argv, "-ngl", "--n-gpu-layers")
    offloaded = total if ngl_text in (None, "all", "auto") else min(int(ngl_text), total)
    if fit:
        log.line(templates["fit_start"])
        if not context_text and os.environ.get("FAKE_FIT_CTX"):
            cut = int(os.environ["FAKE_FIT_CTX"])
            log.line(templates["fit_context"].format(before=pool, after=cut, mib=pool - cut))
            pool = cut
        if ngl_text is None and os.environ.get("FAKE_FIT_LAYERS"):
            offloaded = min(int(os.environ["FAKE_FIT_LAYERS"]), total)
            moe = os.environ.get("FAKE_FIT_MOE")
            expert_layers = int(moe or 0)
            log.line(
                templates["fit_summary"].format(
                    layers=offloaded, overflow="", used=4770, free=17598
                )
            )
            log.line(templates["fit_layers"].format(kind="dense-only" if moe else "dense"))
            log.line(templates["fit_set"].format(layers=offloaded))
            overflow = f" ({int(moe) + 1:>2} overflowing)" if moe else ""
            log.line(
                templates["fit_summary"].format(
                    layers=offloaded, overflow=overflow, used=20000, free=500
                )
            )
        log.line(templates["fit_done"])
    held = os.environ.get("FAKE_HELD_OOM_BELOW")
    if held and not fit and int(option(argv, "--n-cpu-moe") or 0) < int(held):
        log.line(
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 82887.73 MiB on device 0: "
            "cudaMalloc failed: out of memory"
        )
        sys.exit("llama_model_load: error loading model: unable to allocate CUDA0 buffer")
    room = os.environ.get("FAKE_MTP_FIT_CTX")
    if room and option(argv, "--spec-type") == "draft-mtp":
        if fit and not context_text:
            pool = min(pool, int(room))
        elif pool > int(room):
            log.line(
                "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3072.00 MiB on device 0: "
                "cudaMalloc failed: out of memory"
            )
            sys.exit("common_init_from_params: failed to create context with model")
    size = os.path.getsize(model)
    log.line(templates["file_size"].format(gib=size / 2**30))
    log.line(templates["load_mode"])
    log.line(templates["offloaded"].format(n=offloaded, m=total))
    on_device, host_mib, device_read, host_read = place_tensors(meta, offloaded, cpu, expert_layers)
    log.line(templates["model_buffer"].format(device="CPU" if cpu else "CPU_Mapped", mib=host_mib))
    if on_device:
        log.line(
            templates["model_buffer"].format(
                device=gpu + "_Mapped" if gpu == "MTL0" else gpu, mib=on_device
            )
        )
    slot = pool if unified else pool // parallel
    requested = option(argv, "-fa", "--flash-attn") or "auto"
    flash = {"on": "enabled", "off": "disabled"}.get(requested, requested)
    for line in templates["context"]:
        log.line(
            line.format(
                flash=flash,
                parallel=parallel,
                pool=pool,
                slot=slot,
                unified=str(unified).lower(),
                ubatch=ubatch,
            )
        )
    kind = "q8_0" if option(argv, "-ctk", "--cache-type-k") == "q8_0" else "f16"
    kv_mib = pool * layers * 2 * meta["kv_width"] * (1.0625 if kind == "q8_0" else 2) / MIB
    log.line(
        templates["output_buffer"].format(
            device="CPU" if cpu or gpu == "MTL0" else "CUDA_Host", mib=2.32
        )
    )
    host_share = max(total - offloaded - 1, 0) / layers if not cpu else 1.0
    if host_share:
        log.line(templates["kv_buffer"].format(device="CPU", mib=kv_mib * host_share))
    if host_share < 1:
        log.line(templates["kv_buffer"].format(device=gpu, mib=kv_mib * (1 - host_share)))
    cells = pool if unified else slot
    seqs = f"{parallel}/1" if unified else f"{parallel}/{parallel}"
    log.line(templates["kv_size"].format(
        mib=kv_mib, cells=cells, layers=layers, seqs=seqs, kind=kind, half=kv_mib / 2
    ))  # fmt: skip
    if flash == "auto":
        log.line(templates["flash"])
    compute = "CPU" if cpu else gpu
    if apple and os.environ.get("FAKE_METAL_PRIVATE") == "1":
        compute = gpu + "_Private"
    log.line(templates["compute_buffer"].format(device=compute, mib=208.24 * ubatch / 512))
    if apple:
        log.line(templates["graph"])
    log.line(templates["slots"].format(parallel=parallel, slot=slot, unified=str(unified).lower()))
    if os.environ.get("FAKE_SNAPSHOTS"):
        spacing = option(argv, "-cms", "--checkpoint-min-step") or "8192"
        log.line(
            f"srv    load_model: context checkpoints enabled, max = 32, min spacing = {spacing}"
        )
    paced = None
    link_s = 0.0
    if os.environ.get("FAKE_PCIE_GBS") and not cpu:
        streamed = host_mib - embedding_mib(meta)
        link_s = streamed * MIB / (float(os.environ["FAKE_PCIE_GBS"]) * 1e9)
    if os.environ.get("FAKE_BW_HOST_GBS"):
        dev_bw = float(os.environ["FAKE_BW_DEV_GBS"]) * 1e9
        host_bw = float(os.environ["FAKE_BW_HOST_GBS"]) * 1e9
        paced = device_read * MIB / dev_bw + host_read * MIB / host_bw
    return pool, unified, paced, link_s, ubatch


def main() -> None:
    argv = sys.argv[1:]
    if "--version" in argv:
        text = (HERE / "llamacpp_b11669_version.txt").read_text()
        print(text.replace("build 11669", f"build {BUILD_NUMBER}"), end="", file=sys.stderr)
        return
    if "--list-devices" in argv:
        print("Available devices:")
        if os.environ.get("FAKE_BACKEND") == "apple":
            working_set = int(os.environ.get("FAKE_METAL_WS_MIB", "18186"))
            print(f"  MTL0: Apple M4 ({working_set} MiB, {working_set} MiB free)")
            print("  BLAS: Accelerate (0 MiB, 0 MiB free)")
        else:
            print("  (none)")
        return
    if "--help" in argv or "-h" in argv:
        dropped = set(filter(None, os.environ.get("FAKE_LLAMA_HELP_DROP", "").split(",")))
        print(help_text(dropped), end="")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fake_census

    fake_census.start(os.environ.get("GGML_CUDA_DISABLE_GRAPHS") == "1")
    model = option(argv, "-m", "--model") or ""
    meta = read_gguf_meta(model)
    if meta is None:
        sys.exit("llama_model_load: error loading model: failed to load model")
    log = Log()
    if launches := os.environ.get("FAKE_LAUNCH_LOG"):
        with open(launches, "a", encoding="utf-8") as launched:
            launched.write(f"launch {os.environ.get('GLIBC_TUNABLES', '')}\n")
    pool, unified, paced, link_s, ubatch = startup_log(log, argv, meta, model)
    parallel_text = option(argv, "-np", "--parallel")
    parallel = int(parallel_text) if parallel_text and int(parallel_text) > 0 else DEFAULT_PARALLEL
    cache = "--no-cache-prompt" not in argv
    alias = option(argv, "-a", "--alias") or Path(model).name
    port = int(option(argv, "--port") or 8080)
    host = option(argv, "--host") or "127.0.0.1"
    state = State(alias, parallel, pool or DEFAULT_CONTEXT, unified, cache, model)
    state.log = log
    state.paced_step_s = paced
    state.link_s = link_s
    state.ubatch = ubatch
    state.reworded = os.environ.get("FAKE_REWORD_FLAG") in argv
    if os.environ.get("FAKE_FEWER_CALLS_FLAG") in argv:
        state.fewer_calls = int(os.environ.get("FAKE_FEWER_CALLS_EVERY", "3"))
    if os.environ.get("FAKE_FAST_FLAG") in argv:
        state.speed = 0.6
    state.far = os.environ.get("FAKE_FAR_FLAG") in argv
    if os.environ.get("FAKE_FLIP_FLAG") in argv or state.far:
        state.flip_every = int(os.environ["FAKE_FLIP_EVERY"])
    state.wrong = os.environ.get("FAKE_WRONG_FLAG") in argv
    state.spacing = int(option(argv, "-cms", "--checkpoint-min-step") or 8192)
    state.picks_by_similarity = option(argv, "-sps", "--slot-prompt-similarity") != "0"
    state.fits = 7 if int(option(argv, "-b", "--batch-size") or 2048) >= 2048 else 4
    server = BurstServer((host, port), make_handler(state))
    loaded = threading.Timer(
        max(state.ready_at - time.monotonic(), 0.0), lambda: log.line(LOG_TEMPLATES["loaded"])
    )
    loaded.daemon = True
    loaded.start()
    log.line(LOG_TEMPLATES["listening"].format(host=host, port=port))
    threading.Thread(target=exit_with_parent, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
