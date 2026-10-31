"""Two measurements the cost model needs and a log cannot give: how fast the host memory is read
by a decode step, and how fast host weights stream to the device while prefilling.

The host bandwidth comes from one extra launch with ``-ngl 0`` and is cached per host, binary,
model and thread count. The PCIe rate
comes from two runs the ladder already makes, the baseline and the ``-ub`` raise: with op-offload
the host weights cross the link once per ubatch, so the time a long prompt saves per ubatch
fewer is the time those weights take to cross.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import shutil
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ...anatomy import ModelAnatomy
from ...files import cache_dir, write_json
from ...platforms.cpu import HostFacts, host_facts
from ...platforms.nvidia import NVIDIA, pcie_gbs_theoretical
from ...settings import Settings
from ..protocol import ProbeResult, ProbeRun
from .placement import Placement, Prefill, kind_of
from .startup import read_startup

if TYPE_CHECKING:
    from ...runtime import Runtime, ServeSpec

PROBE_CACHE = "~/.cache/tensward/llamacpp/probes.json"
PROBE_TOKENS = 64
MIN_PROBE_TOKENS = 16
WARM_TOKENS = 8
PROMPT = "Count slowly from one to one hundred, one number per line."
MIN_SLOPE_PROMPT = 2048  # below it one ubatch holds the whole prompt
OFFLOAD_MIN_BATCH = 32  # ggml-cuda runs the ops of a batch this small on the host
HOST_BANDWIDTH_FACT = "host.mem_bw_gbs"
PCIE_RATE_FACT = "pcie_gbs"  # (GB/s, where it comes from), in the run's resolved facts
SINGLE_REQUEST_BATCH = 1.2  # the average running batch up to which a step is one request
PROBE_CONTEXT = 2048  # the probe sends 8 and 64 tokens; a big pool would only fill host RAM
SIZE_FLAGS = ("--ctx-size", "--parallel", "--kv-unified", "--no-kv-unified")
PLACEMENT_FLAGS = ("--n-cpu-moe", "--cpu-moe", "--override-tensor", "--n-cpu-ffn")


class ProbeFailed(Exception):
    """The host probe could not measure; the message says why."""


@dataclass(frozen=True, slots=True)
class BandwidthProbe:
    mem_bw_gbs: float
    threads: int
    tokens: int
    cached: bool


@dataclass(frozen=True, slots=True)
class PcieProbe:
    gbs_effective: float | None
    ub_small: int
    ub_large: int
    theoretical_gbs: float | None
    reason: str
    gen: int | None = None
    width: int | None = None
    stored_gbs: float | None = None  # a rate an earlier run on this host and link measured


@dataclass(frozen=True, slots=True)
class PrefillPoint:
    ubatch: int
    prompt_tokens: int
    prompt_ms: float  # median server timings.prompt_ms of the long prompts it computed


def cache_path() -> Path:
    return cache_dir() / "llamacpp" / "probes.json"


def host_fingerprint(host: HostFacts, cpu_model: str) -> str:
    facts = (
        cpu_model, host.logical_cpus, host.physical_cores, host.sockets, host.numa_nodes,
        round(host.mem_gib), sorted(host.isa),
    )  # fmt: skip
    return hashlib.sha256(repr(facts).encode()).hexdigest()


def binary_digest(runtime: Runtime) -> str:
    """The SHA-256 of the server binary a local runtime runs, else of the runtime's target text
    (a container image)."""
    command = getattr(runtime, "command", None)
    path = shutil.which(command[0]) if command else None
    if path is None:
        return hashlib.sha256(runtime.target.encode()).hexdigest()
    digest = hashlib.sha256()
    here = Path(path).resolve()
    libraries = sorted([*here.parent.glob("libggml*"), *here.parent.glob("libllama*")])
    try:
        for file in (here, *(lib.resolve() for lib in libraries if lib.is_file())):
            with open(file, "rb") as binary:
                while block := binary.read(1 << 20):
                    digest.update(block)
    except OSError:
        return hashlib.sha256(path.encode()).hexdigest()
    return digest.hexdigest()


def model_fingerprint(anatomy: ModelAnatomy | None) -> str:
    """What makes one model's host reads differ from another's: its architecture and the bytes
    its tensors take."""
    if anatomy is None:
        return "unknown"
    facts = (anatomy.architecture, anatomy.layers, sorted(anatomy.weight_bytes_by_role.items()))
    return hashlib.sha256(repr(facts).encode()).hexdigest()[:16]


def cache_key(host_fp: str, binary: str, model: str, threads: int | None) -> str:
    return f"{host_fp}:{binary}:{model}:{threads or 'default'}"


def _stored() -> dict[str, Any]:
    try:
        found = json.loads(cache_path().read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _remember(key: str, probe: BandwidthProbe, method: str) -> None:
    stored = _stored()
    stored[key] = dataclasses.asdict(probe) | {"cached": False, "method": method}
    try:
        cache_path().parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_path(), stored)
    except OSError:
        pass


def cached_bandwidth(key: str) -> BandwidthProbe | None:
    """The probe a launch stored for ``key``."""
    found = _stored().get(key)
    if not isinstance(found, dict) or found.get("method", "launch") != "launch":
        return None
    try:
        return BandwidthProbe(
            mem_bw_gbs=float(found["mem_bw_gbs"]),
            threads=int(found["threads"]),
            tokens=int(found["tokens"]),
            cached=True,
        )
    except (TypeError, KeyError, ValueError):
        return None


def launched_bandwidth(
    host: HostFacts, runtime: Runtime, model: str, threads: int | None
) -> float | None:
    """The host memory bandwidth a launched probe stored for this host, binary and model: the
    one at ``threads`` if there is one, else the highest at any thread count."""
    fingerprint, binary = host_fingerprint(host, _cpu_model()), binary_digest(runtime)
    if found := cached_bandwidth(cache_key(fingerprint, binary, model, threads)):
        return found.mem_bw_gbs
    prefix = f"{fingerprint}:{binary}:{model}:"
    stored = (cached_bandwidth(key) for key in _stored() if key.startswith(prefix))
    return max((found.mem_bw_gbs for found in stored if found), default=None)


def pcie_key(host_fp: str, gpu: str, link: tuple[int, int] | None) -> str:
    return f"pcie:{host_fp}:{gpu}:{'x'.join(map(str, link)) if link else 'unknown'}"


def stored_pcie(key: str) -> float | None:
    found = _stored().get(key)
    rate = found.get("gbs") if isinstance(found, dict) else None
    return float(rate) if isinstance(rate, (int, float)) and rate > 0 else None


def _remember_pcie(key: str, gbs: float) -> None:
    stored = _stored()
    stored[key] = {"gbs": gbs}
    try:
        cache_path().parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_path(), stored)
    except OSError:
        pass


def read_bytes(p: Placement) -> int:
    """Bytes one decode step reads when every layer is in host memory."""
    return p.device_bytes_per_token + p.host_bytes_per_token


def host_settings(settings: Settings) -> Settings:
    """The baseline's settings with every layer on the host."""
    extra = {k: v for k, v in settings.extra_args.items() if k not in PLACEMENT_FLAGS + SIZE_FLAGS}
    extra["--n-gpu-layers"] = "0"
    return dataclasses.replace(
        settings, extra_args=extra, max_context_len=PROBE_CONTEXT, max_concurrent_requests=1
    )


def passive_bandwidth(
    p: Placement, tpot_ms: float | None, running: float | None
) -> BandwidthProbe | None:
    """A floor for the host bandwidth read off a CPU-only baseline, where every step reads all
    the weights: only off a single-request step, and never stored, because it holds everything
    else the step did."""
    if not tpot_ms or tpot_ms <= 0 or running is None or running > SINGLE_REQUEST_BATCH:
        return None
    return BandwidthProbe(
        mem_bw_gbs=read_bytes(p) / (tpot_ms * 1e6), threads=0, tokens=0, cached=False
    )


async def _final_timings(
    client: httpx.AsyncClient, url: str, model: str, tokens: int
) -> Mapping[str, float] | None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    found: Mapping[str, float] | None = None
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as response:
        if response.status_code != 200:
            return None
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                frame = json.loads(line[6:])
            except ValueError:
                continue
            if isinstance(frame.get("timings"), dict):
                found = frame["timings"]
    return found


async def _decode_rate(
    runtime: Runtime, spec: ServeSpec, ready_timeout_s: float
) -> tuple[float, int, int] | None:
    from ...runtime import wait_until_ready  # runtime imports the engines

    server = runtime.start(spec)
    try:
        await wait_until_ready(
            server, spec.engine, api_key=spec.api_key,
            served_model_name=spec.served_model_name, timeout_s=ready_timeout_s,
            interval_s=0.1,
        )  # fmt: skip
        headers = {"Authorization": f"Bearer {spec.api_key}"}
        model = spec.engine.request_model(spec.served_model_name)
        async with httpx.AsyncClient(headers=headers, timeout=600.0) as client:
            await _final_timings(client, server.endpoint_url, model, WARM_TOKENS)
            found = await _final_timings(client, server.endpoint_url, model, PROBE_TOKENS)
        log = server.log_text()
    finally:
        server.stop()
    if not found:
        return None
    count, millis = found.get("predicted_n", 0), found.get("predicted_ms", 0)
    if count < MIN_PROBE_TOKENS or millis <= 0:
        return None
    return count / (millis / 1000), int(count), read_startup(log).threads or 0


def host_bandwidth(
    runtime: Runtime, spec: ServeSpec, p: Placement, key: str, ready_timeout_s: float = 600.0
) -> BandwidthProbe | None:
    """The bandwidth a decode step reads host memory at: one more launch of ``spec`` (the
    baseline's) with every layer on the host, asked for 64 tokens at one request with its prompt
    already cached. ``key`` is the cache key. None when the launch fails or answers too few
    tokens."""
    from ...runtime import generate_api_key, select_loopback_port  # runtime imports the engines

    spec = dataclasses.replace(
        spec, settings=host_settings(spec.settings), port=select_loopback_port(),
        api_key=generate_api_key(),
    )  # fmt: skip
    try:
        measured = asyncio.run(_decode_rate(runtime, spec, ready_timeout_s))
    except Exception as error:  # noqa: BLE001 - a probe never fails the analysis
        raise ProbeFailed(f"the launch failed: {error}".splitlines()[0]) from error
    if measured is None:
        raise ProbeFailed("the server answered too few tokens to time")
    rate, tokens, threads = measured
    probe = BandwidthProbe(
        mem_bw_gbs=read_bytes(p) * rate / 1e9, threads=threads, tokens=tokens, cached=False
    )
    _remember(key, probe, "launch")
    return probe


def prefill_point(rows: Sequence[Mapping[str, Any]], ubatch: int) -> PrefillPoint | None:
    """The median prefill of the long prompts a run computed: successful requests with 2,048 or
    more computed prompt tokens (a cached prefix is not computed, and cancels between two runs
    of the same workload)."""
    times, tokens = [], []
    for row in rows:
        timings = row.get("timings") or {}
        computed = timings.get("prompt_n", 0)
        if (
            row.get("outcome") == "success"
            and "prompt_ms" in timings
            and computed >= MIN_SLOPE_PROMPT
        ):
            times.append(timings["prompt_ms"])
            tokens.append(computed)
    if not times or not ubatch:
        return None
    return PrefillPoint(ubatch, round(statistics.median(tokens)), statistics.median(times))


def _streaming_ubatches(prompt_n: int, ubatch: int) -> int:
    """The ubatches of a computed prompt that stream the host weights: ggml-cuda offloads an
    op only for a batch of OFFLOAD_MIN_BATCH tokens or more."""
    full, rest = divmod(prompt_n, ubatch)
    return (full if ubatch >= OFFLOAD_MIN_BATCH else 0) + (rest >= OFFLOAD_MIN_BATCH)


def prefill_work(
    rows: Iterable[Mapping[str, Any]], ubatch: int | None, running: float | None
) -> Prefill | None:
    """The ubatches of the computed prompts that stream host weights, and the decode steps the
    run's generated tokens make at the average running batch. Each prompt counts its own
    ubatches: prompts computed in one ubatch share it, so batching can make the count too high."""
    if not ubatch or not running:
        return None
    ubatches = generated = 0
    for row in rows:
        timings = row.get("timings") or {}
        if row.get("outcome") == "success":
            ubatches += _streaming_ubatches(timings.get("prompt_n", 0), ubatch)
            generated += timings.get("predicted_n", 0)
    return Prefill(ubatches, generated / running) if ubatches and generated else None


def pcie_slope(
    baseline: PrefillPoint | None,
    step: PrefillPoint | None,
    p: Placement,
    link: tuple[int, int] | None,
) -> PcieProbe:
    """The rate host weights cross PCIe at, from two runs of the same long prompts at different
    ``-ub``: ``host_weight_bytes * (n_small - n_large) / (t_small - t_large)`` with ``n`` the
    ubatches a prompt takes. ``gbs_effective`` is None, with the reason, when the runs do not
    show a slope; ``step`` is None until the raise-ubatch step has run."""
    theoretical = pcie_gbs_theoretical(*link) if link else None
    small = baseline.ubatch if baseline else 0
    large = 0

    def none(reason: str) -> PcieProbe:
        return PcieProbe(None, small, large, theoretical, reason, *(link or (None, None)))

    if baseline is None:
        return none("no baseline run computed prompts of 2,048 tokens or more")
    if step is None:
        return none("the raise-ubatch step has not run")
    if not p.host_weight_bytes:
        return none("no weights are in host memory")
    low, high = sorted((baseline, step), key=lambda point: point.ubatch)
    small, large = low.ubatch, high.ubatch
    if small == large:
        return none("both runs used the same -ub")
    if abs(low.prompt_tokens - high.prompt_tokens) > 0.02 * low.prompt_tokens:
        return none("the two runs computed prompts of different lengths")
    saved = math.ceil(low.prompt_tokens / small) - math.ceil(low.prompt_tokens / large)
    seconds = (low.prompt_ms - high.prompt_ms) / 1000
    if saved <= 0:
        return none("one ubatch holds the whole prompt at both sizes")
    if seconds <= 0:
        return none("the larger ubatch did not shorten prefill: host weights are not streamed")
    rate = p.host_weight_bytes * saved / seconds / 1e9
    return PcieProbe(rate, small, large, theoretical, "", *(link or (None, None)))


def _cpu_model() -> str:
    try:
        lines = Path("/proc/cpuinfo").read_text("utf-8", "replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip() in ("model name", "Model", "Hardware"):
            return value.strip()
    return ""


def _baseline_prefill(project_dir: Path, snapshot_id: str) -> PrefillPoint | None:
    """The prefill point of the newest run of this setup that changed no engine argument."""
    from ...runs import baseline_runs  # runs imports the engines

    for run_dir in baseline_runs(project_dir, snapshot_id):
        try:
            found = (json.loads((run_dir / "run.json").read_text("utf-8")).get("probes") or {}).get(
                "prefill"
            )
        except (OSError, ValueError):
            continue
        return PrefillPoint(**found) if found else None
    return None


def _host_probe(
    run: ProbeRun, p: Placement, threads: int | None
) -> tuple[dict[str, Any] | None, str, bool]:
    """(the probe as a dict, why there is none, whether it was launched or cached from a
    launch). A CPU-only launch whose probe launch failed falls back to a floor read off its own
    step when that was one request; that reading is reported but is not a bandwidth to compute
    this run's ceiling from."""
    host = run.host or host_facts()
    key = cache_key(
        host_fingerprint(host, _cpu_model()),
        binary_digest(run.runtime),
        model_fingerprint(run.anatomy),
        threads,
    )
    if (stored := cached_bandwidth(key)) is not None:
        return dataclasses.asdict(stored) | {"method": "launch"}, "", True
    try:
        probe = host_bandwidth(run.runtime, run.spec, p, key, run.ready_timeout_s)
    except ProbeFailed as error:
        passive = (
            passive_bandwidth(p, run.measurement.tpot_p50_ms, run.measurement.mean_running)
            if kind_of(p) == "cpu_only"
            else None
        )
        if passive is None:
            return None, str(error), False
        return dataclasses.asdict(passive) | {"method": "passive"}, "", False
    if probe is None:
        return None, "the decode step time is unknown", False
    return dataclasses.asdict(probe) | {"method": "launch"}, "", True


def _pcie_rate(pcie: Mapping[str, Any] | None) -> tuple[float, str] | None:
    """The best rate a PCIe probe block knows and what it rests on: this run's slope, one an
    earlier run on this host measured, else the link's theoretical rate, which the real one
    stays below."""
    if not pcie:
        return None
    if pcie.get("gbs_effective"):
        return pcie["gbs_effective"], "measured"
    if pcie.get("stored_gbs"):
        return pcie["stored_gbs"], "measured by an earlier run on this host"
    if pcie.get("theoretical_gbs"):
        return pcie["theoretical_gbs"], "the link's theoretical rate"
    return None


def probe_run(run: ProbeRun) -> ProbeResult | None:
    """The probes of a finished baseline or step run, where its placement puts weights in host
    memory (or the KV cache): the host bandwidth, the PCIe rate and the prefill point a later
    run compares against. None for a launch with everything on the device."""
    found = run.resolved.facts.get("placement") if run.resolved else None
    if not found:
        return None
    p = Placement(**found)
    kind = kind_of(p)
    if kind == "full_gpu":
        return None
    started = read_startup(run.server_log)
    block: dict[str, Any] = {"host": None, "pcie": None, "prefill": None}
    gbs = None
    if run.launch:
        block["host"], reason, launched = _host_probe(run, p, started.threads)
        if reason:
            block["host_reason"] = reason
        gbs = block["host"]["mem_bw_gbs"] if block["host"] and launched else None
    else:
        block["host_reason"] = "probes are off (--no-probes)"
    if run.platform == "nvidia" and p.host_weight_bytes and kind != "cpu_only":
        gpus = getattr(run.runtime, "gpus", None)
        link = NVIDIA.pcie_link(gpus[0] if gpus else 0)
        point = prefill_point(run.rows, started.ubatch or 0)
        block["prefill"] = dataclasses.asdict(point) if point else None
        if point is None:
            slope = PcieProbe(
                None, started.ubatch or 0, 0,
                pcie_gbs_theoretical(*link) if link else None,
                "no prompt of 2,048 tokens or more was computed whole (one ubatch holds a "
                "shorter prompt)",
                *(link or (None, None)),
            )  # fmt: skip
        elif run.engine_args:
            slope = pcie_slope(_baseline_prefill(run.project_dir, run.snapshot_id), point, p, link)
        else:
            slope = pcie_slope(point, None, p, link)
        devices = run.runtime.platform.detect()
        first = str(gpus[0]) if gpus else "0"
        name = devices[int(first)].name if first.isdigit() and int(first) < len(devices) else ""
        key = pcie_key(host_fingerprint(run.host or host_facts(), _cpu_model()), name, link)
        if slope.gbs_effective:
            _remember_pcie(key, slope.gbs_effective)
        elif (earlier := stored_pcie(key)) is not None:
            slope = dataclasses.replace(slope, stored_gbs=earlier)
        block["pcie"] = dataclasses.asdict(slope)
    resolved = run.resolved
    extra: dict[str, Any] = {}
    if gbs:
        extra[HOST_BANDWIDTH_FACT] = gbs
    if rate := _pcie_rate(block["pcie"]):
        extra[PCIE_RATE_FACT] = rate
    if extra and resolved is not None:
        resolved = dataclasses.replace(resolved, facts={**resolved.facts, **extra})
    return ProbeResult(block, gbs, resolved)
