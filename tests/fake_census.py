"""A stand-in for the census library, used by the fake engine servers (stdlib only).

``start()`` does what the injected library does in a real server, and nothing when the
environment carries no ``CUDA_INJECTION64_PATH`` or the file is missing: it maps the injection
file and the first ``libcupti.so.*`` on ``LD_LIBRARY_PATH`` (so ``/proc/<pid>/maps`` shows both),
prints the ``twcensus: on`` line a real 2.1 library prints (``twcensus: off: <reason>`` when its
output folder is not writable), and runs a thread that reacts to ``<prefix>.off`` once a second
and rewrites ``<out>.<pid>.json`` and ``<out>.<pid>.tl.jsonl``.

Kernels are counted only while ``.off`` is absent; the snapshot says ``paused`` while it exists.
The timeline holds buckets of the unpaused interval only. Knobs: ``FAKE_CENSUS_SMS`` (the
multiprocessor count, 58), ``FAKE_CENSUS_ZERO=1`` (no kernel is ever recorded),
``FAKE_CENSUS_EXIT_HANG=1`` (SIGTERM is ignored), ``FAKE_CENSUS_CRASH_AFTER=<s>`` (abort after
that long), ``FAKE_CENSUS_START_DELAY_S`` (the start stalls this long before the server runs),
``FAKE_CENSUS_READ_ONLY=1`` (the output folder refuses writes), ``FAKE_CENSUS_H2D_PER_KERNEL``
(bytes copied to the device per kernel, 1000), ``FAKE_CENSUS_COPY_SHARE`` (the share of each
timeline bucket a copy runs with no kernel, 0), ``FAKE_CENSUS_ENV_LOG=<path>``
(every start, with or without the census, appends its census and graph variables).

With ``engine_child``, as in vLLM, the server forks an engine process that records while the
server itself only maps the library.
"""

from __future__ import annotations

import errno
import json
import mmap
import os
import re
import signal
import threading
import time
from pathlib import Path

VERSION = "2.1"
BUCKET_NS = 100_000_000
KERNELS_PER_S = 3000
KERNEL_NS = 20_000
KERNEL_NAME = "void fake_gemm_kernel<128, 4>(float const*, float const*, float*, int)"
CUPTI_RUN = {"cu12": 27, "cu13": 130001}
CUPTI_BUILD = {"cu12": 22, "cu13": 130000}
MAPPED: list[mmap.mmap] = []


def _map(path: Path) -> None:
    with path.open("rb") as handle:
        MAPPED.append(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))


def _cupti_file() -> Path | None:
    for folder in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if folder and (found := sorted(Path(folder).glob("libcupti.so.*"))):
            return found[0]
    return None


class Recorder:
    def __init__(self, kind: str, prefix: str, graphs_off: bool, flush: str) -> None:
        self.kind = kind
        self.control = prefix
        self.prefix = f"{prefix}.{os.getpid()}"
        self.graphs_off = graphs_off
        self.flush = flush
        self.sms = int(os.environ.get("FAKE_CENSUS_SMS", "58"))
        self.zero = os.environ.get("FAKE_CENSUS_ZERO") == "1"
        self.kernels = 0
        self.copy_bytes = 0
        self.h2d_per_kernel = int(os.environ.get("FAKE_CENSUS_H2D_PER_KERNEL", "1000"))
        self.copy_share = float(os.environ.get("FAKE_CENSUS_COPY_SHARE", "0"))
        self.first_ns = 0
        self.last_ns = 0
        self.paused = Path(f"{prefix}.off").exists()
        self.timeline = Path(f"{self.prefix}.tl.jsonl")
        self.bucket_ns = 0

    def snapshot(self) -> dict:
        n = 0 if self.zero else self.kernels
        busy = n * KERNEL_NS * 6 // 10
        return {
            "census": VERSION,
            "names_overflow": 0,
            "kinds": ["kernels", "memcpy", "sync", "memory"],
            "pid": os.getpid(),
            "final": False,
            "t_first_ns": self.first_ns,
            "t_last_ns": self.last_ns,
            "busy_union_ns": busy,
            "n_kernels": n,
            "n_records": n,
            "dropped": 0,
            "late": 0,
            "late_ns": 0,
            "pending": 0,
            "flushes": 0,
            "paused": self.paused,
            "api": False,
            "sync": True,
            "now_ns": time.monotonic_ns(),
            "gaps": [{"n": n, "ns": n * 1000}] + [{"n": 0, "ns": 0}] * 6,
            "copies": {"1": {"n": n // 100, "ns": n * 10, "bytes": self.copy_bytes}},
            "syncs": {"3": {"n": n // 50, "ns": n * 100}},
            "apis": [],
            "kernels": [self.kernel_row(n)] if n else [],
            "lib": {
                "kind": self.kind,
                "src": "0" * 16,
                "cupti_build": CUPTI_BUILD[self.kind],
                "cupti_run": CUPTI_RUN[self.kind],
                "graph": "kernel",
                "pinned": False,
                "sm_count": self.sms,
            },
            "graph_traces": {"n": 0, "ns": 0},
            "copies_detail": {"1:1:3": {"n": n // 100, "ns": n * 10, "bytes": self.copy_bytes}},
            "memory": {
                "alloc_n": 12,
                "alloc_bytes": 1 << 30,
                "free_n": 2,
                "free_bytes": 1 << 20,
                "peak_bytes": 1 << 30,
                "by_kind": {"3": 1 << 30},
            },
        }

    def kernel_row(self, n: int) -> dict:
        return {
            "name": KERNEL_NAME,
            "id": -1,
            "n": n,
            "ns": n * KERNEL_NS,
            "min_ns": KERNEL_NS // 2,
            "max_ns": KERNEL_NS * 2,
            "blocks": n * 64,
            "blocks_min": 64,
            "blocks_max": 64,
            "tpb": 256,
            "tpb_max": 256,
            "regs": 64,
            "shm_static": 0,
            "shm_dynamic_max": 0,
            "graph_n": 0 if self.graphs_off else n,
            "queue_ns": 0,
        }

    def write_snapshot(self) -> None:
        temporary = Path(f"{self.prefix}.json.tmp")
        temporary.write_text(json.dumps(self.snapshot()))
        os.replace(temporary, f"{self.prefix}.json")

    def record_interval(self, start_ns: int, end_ns: int) -> None:
        rate = 0 if self.zero else KERNELS_PER_S
        with self.timeline.open("a", encoding="utf-8") as handle:
            while self.bucket_ns + BUCKET_NS <= end_ns:
                if self.bucket_ns >= start_ns:
                    kernels = rate * BUCKET_NS // 10**9
                    row = {
                        "t_ns": self.bucket_ns,
                        "busy_ns": kernels * KERNEL_NS * 6 // 10,
                        "kernels": kernels,
                        "queue_ns": 0,
                        "h2d": kernels * self.h2d_per_kernel,
                        "d2h": 0,
                        "copy_ns": int(BUCKET_NS * self.copy_share),
                        "sync_ns": 0,
                    }
                    handle.write(json.dumps(row) + "\n")
                self.bucket_ns += BUCKET_NS

    def run(self) -> None:
        self.timeline.write_text(json.dumps({"census": VERSION, "t_ns": 0}) + "\n")
        self.bucket_ns = time.monotonic_ns() // BUCKET_NS * BUCKET_NS
        last = time.monotonic_ns()
        self.write_snapshot()
        while True:
            time.sleep(1.0)
            now = time.monotonic_ns()
            was_paused = self.paused
            self.paused = Path(f"{self.control}.off").exists()
            if not was_paused:
                elapsed = now - last
                new = elapsed * KERNELS_PER_S // 10**9
                self.kernels += new
                self.copy_bytes += new * self.h2d_per_kernel
                self.first_ns = self.first_ns or last
                self.last_ns = now
                self.record_interval(last, now)
            else:
                self.bucket_ns = now // BUCKET_NS * BUCKET_NS
            last = now
            self.write_snapshot()


ENV_LOGGED = ("CUDA_INJECTION64_PATH", "TWCENSUS_OUT", "GGML_CUDA_DISABLE_GRAPHS")


def start(graphs_off: bool, engine_child: bool = False) -> None:
    if log := os.environ.get("FAKE_CENSUS_ENV_LOG"):
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({name: os.environ.get(name) for name in ENV_LOGGED}) + "\n")
    library = os.environ.get("CUDA_INJECTION64_PATH")
    if not library or not Path(library).is_file():
        return
    if engine_child:
        _map(Path(library))
        if (cupti := _cupti_file()) is not None:
            _map(cupti)
        server = os.getpid()
        if os.fork():
            time.sleep(float(os.environ.get("FAKE_CENSUS_START_DELAY_S", "0")))
            return
        _record(library, graphs_off)
        while os.getppid() == server:
            time.sleep(1.0)
        os._exit(0)
    _record(library, graphs_off)
    time.sleep(float(os.environ.get("FAKE_CENSUS_START_DELAY_S", "0")))


def _record(library: str, graphs_off: bool) -> None:
    match = re.search(r"-(cu\d+)\.so$", library)
    kind = match.group(1) if match else "cu12"
    if kind not in CUPTI_RUN:
        kind = "cu12"
    prefix = os.environ.get("TWCENSUS_OUT", "")
    named = f"{prefix}.{os.getpid()}"
    try:
        if os.environ.get("FAKE_CENSUS_READ_ONLY") == "1":
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES))
        Path(f"{named}.probe").write_text("")
        Path(f"{named}.probe").unlink()
    except OSError as error:
        print(f"twcensus: off: output {named}: {error.strerror}", flush=True)
        return
    _map(Path(library))
    if (cupti := _cupti_file()) is not None:
        _map(cupti)
    flush = os.environ.get("TWCENSUS_FLUSH", "ondemand")
    recorder = Recorder(kind, prefix, graphs_off, flush)
    state = " (paused)" if recorder.paused else ""
    print(
        f"twcensus: on, pid {os.getpid()}, out {named}{state}, flush {flush}, census {VERSION}, "
        f"timeline cap 16 MiB, kind {kind}, cupti {CUPTI_RUN[kind]}/{CUPTI_BUILD[kind]}, "
        "graph kernel, pinned 0",
        flush=True,
    )
    threading.Thread(target=recorder.run, daemon=True).start()
    if os.environ.get("FAKE_CENSUS_EXIT_HANG") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if after := os.environ.get("FAKE_CENSUS_CRASH_AFTER"):
        timer = threading.Timer(float(after), os.abort)
        timer.daemon = True
        timer.start()
