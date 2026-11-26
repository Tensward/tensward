"""The census start: after the measurement, Tensward starts the setup once more on its own, with
the GPU census in it, and records a short slice of the workload. It is a diagnostic, never part
of the measurement, and a problem with it never fails the analysis."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .. import runtime as runtimes
from ..context import current as analysis
from ..engines import Engine
from ..engines.vllm import start_phases
from ..errors import AnalyseFailure
from ..images import ImageSource
from ..measure import Live, serving, spec_for, timed_requests, warm_up
from ..profiling.counters import gpu_memory_mib, memory_returned
from ..profiling.run import INTERRUPTED
from ..progress import PhaseStarted, emit
from ..project import ResolvedProject
from ..runtime import ERROR_LINE, ERROR_LINE_CHARS, CensusLaunch, Runtime, ServeSpec
from ..settings import Settings
from . import policy
from .inefficiency import inefficiencies
from .library import census_home
from .proof import NO_OUTPUT, AttachProof, prove_attach
from .readings import ReadingContext, compute_readings
from .record import RECORD_DIR, CensusRecord, write_record
from .snapshot import CensusWindow, read_snapshots, read_timeline, window

CENSUS_LINE = (
    "census: a short separate start that records what the GPU runs (diagnostic, not a speed claim)"
)
OVERRUN = "the census start passed the 10-minute limit and was stopped"
LIBRARY_PATH = "LD_LIBRARY_PATH"


@dataclass(frozen=True, slots=True)
class _Seen:
    proof: AttachProof
    window: CensusWindow
    span_s: float
    error: str = ""


def backend_of(engine: Engine, runtime: Runtime) -> str:
    """The GPU backend the engine is built for: a llama.cpp build names its own (a build that
    names none counts as CUDA, as its libraries decide); vLLM runs on CUDA."""
    if engine.family != "llamacpp":
        return "cuda"
    from ..engines.llamacpp.command import known_build

    built = known_build(runtime.target) or known_build()
    backends = built[0].backends if built else ()
    return "cuda" if not backends or "CUDA" in backends else backends[0].lower()


def _image_library_path(image: str) -> str:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image],
            capture_output=True,
            text=True,
            timeout=runtimes.DOCKER_TIMEOUT_S,
            check=False,
        )
        env = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    for entry in env or ():
        name, _, value = str(entry).partition("=")
        if name == LIBRARY_PATH:
            return value
    return ""


def census_launch(
    plan: policy.CensusPlan, runtime: Runtime, settings: Settings, out_dir: Path
) -> CensusLaunch:
    """The census additions of a start: the engine's CUPTI folder first on the library path,
    the path the engine has otherwise (its setup's, its image's or this environment's) kept."""
    own = settings.extra_env.get(LIBRARY_PATH)
    if own is None:
        own = (
            _image_library_path(runtime.target) if plan.docker else os.environ.get(LIBRARY_PATH, "")
        )
    folders = [plan.cupti.folder, *(f for f in own.split(":") if f and f != plan.cupti.folder)]
    return CensusLaunch(
        library=plan.library.path,
        cupti_dir=plan.cupti.folder,
        ld_library_path=":".join(folders),
        ship_env=dict(plan.profile.ship_env),
        out_dir=out_dir,
    )


async def census_slice(
    project: ResolvedProject, spec: ServeSpec, live: Live, run_id: str, seconds: float,
    req_s: float,
) -> tuple[int, int, int, int, str, int]:  # fmt: skip
    """Run about ``seconds`` of the workload at the measured rate, at least 16 requests, cut at
    twice that; the requests completed, those that failed, the first dispatch and the last end
    (monotonic ns), the first failed request's error and the output tokens of the completed
    ones (0 when the slice was cut)."""
    count = max(16, math.ceil(req_s * seconds))
    began = time.monotonic_ns()
    seen = {"ok": 0, "failed": 0, "last": began}

    def done(finished: int, failed: int) -> None:
        seen["ok"], seen["failed"] = finished - failed, failed
        seen["last"] = time.monotonic_ns()

    try:
        records = await timed_requests(
            project, spec, live, run_id, count, 2 * seconds, "census", verb="recording",
            on_done=done,
        )  # fmt: skip
    except TimeoutError:
        return seen["ok"], seen["failed"], began, seen["last"], "", 0
    if not records:
        return 0, 0, began, began, "", 0
    completed = sum(record.outcome == "success" for record in records)
    error = next((r.error for r in records if r.outcome != "success" and r.error), "")
    tokens = sum(r.committed_output_tokens or 0 for r in records if r.outcome == "success")
    return (
        completed,
        len(records) - completed,
        min(record.dispatch_ns for record in records),
        max(record.terminal_ns for record in records),
        str(error),
        tokens,
    )


async def _sequence(
    project: ResolvedProject, engine: Engine, runtime: Runtime, spec: ServeSpec,
    launch: CensusLaunch, plan: policy.CensusPlan, ready_timeout_s: float, req_s: float,
) -> _Seen:  # fmt: skip
    """Contract 1: warm up while paused, read "before", unpause, record the slice, idle, pause,
    let the library write, read "after" and prove the attach while the server still runs."""
    out, prefix = launch.out_dir, launch.prefix
    off = out / f"{prefix}.off"
    run_id = spec.served_model_name
    async with serving(
        engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=[out],
        phase="census", images=ImageSource(project.images),
    ) as live:  # fmt: skip
        await warm_up(project, spec, live, run_id)
        await asyncio.sleep(policy.IDLE_S)
        before = read_snapshots(out, prefix)
        off.unlink(missing_ok=True)
        await asyncio.sleep(policy.UNPAUSE_WAIT_S)
        completed, failed, start_ns, end_ns, error, tokens = await census_slice(
            project, spec, live, run_id, policy.CENSUS_SLICE_S, req_s
        )
        await asyncio.sleep(policy.IDLE_S)
        off.touch()
        await asyncio.sleep(policy.FLUSH_WAIT_S)
        after = read_snapshots(out, prefix)
        if not live.server.is_running():
            raise AnalyseFailure("the server stopped during the census start")
        proof = prove_attach(
            live.server,
            launch,
            after,
            expected_kind=plan.library.kind,
            graphs_off=bool(plan.profile.ship_env),
        )
    timelines = {
        pid: read_timeline(out / f"{prefix}.{pid}.tl.jsonl", start_ns, end_ns) for pid in after
    }
    found = window(
        before, after, requests=completed, start_ns=start_ns, end_ns=end_ns,
        timelines=timelines, failed=failed, out_tokens=tokens,
    )  # fmt: skip
    return _Seen(proof, found, max(end_ns - start_ns, 0) / 1e9, error)


def _last_error(error: BaseException, out: Path) -> str:
    """The last error line of the census start's log, else the first line of the failure."""
    try:
        log = (out / "server.log").read_text(errors="replace")
    except OSError:
        log = ""
    lines = [line.strip() for line in log.splitlines() if ERROR_LINE.search(line)]
    if lines:
        return lines[-1][:ERROR_LINE_CHARS]
    return str(error).partition("\n")[0] or type(error).__name__


def _group_gone(out: Path) -> bool:
    """Whether nothing of the census start is left: its container, or its process group."""
    try:
        identity = json.loads((out / "serve.json").read_text())["identity"]
    except (OSError, ValueError, KeyError, TypeError):
        return True
    if identity.get("runtime") == "docker":
        return not runtimes.container_running(str(identity.get("container_name")))
    pid = identity.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _judged(
    seen: _Seen, engine: Engine, gpu_model: str | None, graphs_off: bool, version: str | None,
    n_vocab: int | None = None, host_weight_bytes: int | None = None,
) -> CensusRecord:  # fmt: skip
    proof = seen.proof
    if proof.passed():
        sms = {s.lib.get("sm_count") for s in seen.window.after.values() if s.lib}
        sm_count = next(iter(sms)) if len(sms) == 1 else None
        context = ReadingContext(
            engine=engine.name,
            gpu_model=gpu_model,
            sm_count=sm_count if isinstance(sm_count, int) else None,
            graphs_off=graphs_off,
            version=version,
            n_vocab=n_vocab,
            host_weight_bytes=host_weight_bytes,
        )
        found = inefficiencies(seen.window, context)
        return CensusRecord(
            "ok",
            "",
            readings=compute_readings(seen.window, context),
            inefficiency=tuple(found) if found is not None else None,
            requests=seen.window.requests,
            span_s=round(seen.span_s, 3),
            out_tokens=seen.window.out_tokens,
        )
    if proof.attached() and not proof.snapshot_ok:
        return CensusRecord("no_output", NO_OUTPUT)
    failed = seen.window.failed
    if proof.attached() and proof.n_kernels <= 0 and seen.window.requests == 0 and failed:
        why = f": {seen.error}" if seen.error else ""
        return CensusRecord("failed", f"its requests failed ({failed} of {failed}){why}")
    return CensusRecord("not_attached", proof.reason())


def add_census(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    plan: policy.CensusPlan,
    ready_timeout_s: float,
    *,
    req_s: float,
    gpu_model: str | None,
    gpu_index: int,
    version: str | None = None,
) -> CensusRecord:
    """Run the census start and write its record. A census that fails, does not attach or
    writes nothing is a record saying so, never a failure of the analysis; Ctrl-C writes the
    record and is raised again."""
    out = run_dir / RECORD_DIR
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    emit(PhaseStarted(phase="census", detail=CENSUS_LINE))
    began = time.monotonic()
    graphs_off = bool(plan.profile.ship_env)
    launch = census_launch(plan, runtime, settings, out)
    base = spec_for(project, engine, settings, run_dir.name + "-census", None)
    spec = replace(base, census=launch)
    library = {
        "file": plan.library.path.name,
        "kind": plan.library.kind,
        "cupti": plan.cupti.folder,
        "verified": plan.verified,
    }
    context = analysis()
    extra = context.window_extra
    placed = (context.placement or {}).get("placement") or {}
    host_bytes = placed.get("host_weight_bytes")
    cut = extra.get("log_cut")
    interrupted = False
    proof: dict[str, Any] = {}
    cold = gpu_memory_mib(gpu_index)
    budget = (
        policy.CENSUS_CEILING_S
        - runtimes.CENSUS_STOP_GRACE_S
        - policy.SETTLE_S
        - (time.monotonic() - began)
    )
    try:
        seen = asyncio.run(
            asyncio.wait_for(
                _sequence(project, engine, runtime, spec, launch, plan, ready_timeout_s, req_s),
                budget,
            )
        )
        record = _judged(
            seen, engine, gpu_model, graphs_off, version, project.anatomy.vocab,
            int(host_bytes) if host_bytes else None,
        )  # fmt: skip
        proof = asdict(seen.proof)
    except KeyboardInterrupt:
        record, interrupted = CensusRecord("failed", INTERRUPTED), True
    except TimeoutError as error:
        overran = time.monotonic() - began >= budget
        record = CensusRecord("failed", OVERRUN if overran else _last_error(error, out))
    except Exception as error:  # a census problem never fails the analysis
        record = CensusRecord("failed", _last_error(error, out))
    finally:
        if cut is None:
            extra.pop("log_cut", None)
        else:
            extra["log_cut"] = cut
    try:
        gone = _group_gone(out)
        returned, used = memory_returned(cold, gpu_index)
        notes = [record.reason] if record.reason else []
        if not gone:
            notes.append("a process of the census start was still running after it was stopped")
        if cold is not None and used is not None and not returned:
            notes.append(
                "GPU memory did not return to its level before the census start "
                f"({cold} -> {used} MiB)"
            )
        record = replace(
            record,
            reason="; ".join(notes),
            proof={**proof, "group_empty": gone, "memory_mib": {"cold": cold, "after": used}},
            library=library,
            wall_s=round(time.monotonic() - began, 3),
        )
        write_record(run_dir, record)
    except Exception:
        if interrupted:
            with contextlib.suppress(Exception):
                write_record(run_dir, record)
            raise KeyboardInterrupt from None
        raise
    if interrupted:
        raise KeyboardInterrupt
    return record


def _plan(
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    version: str | None,
    load_s: float,
    warmup_s: float,
    wanted: bool,
    built_once: bool = False,
    req_s: float | None = None,
) -> policy.CensusPlan | policy.CensusSkip:
    return policy.plan_census(
        engine,
        runtime,
        version=version,
        platform=runtime.platform,
        backend=backend_of(engine, runtime),
        parallel=engine.parallel_degree(settings),
        load_s=load_s,
        warmup_s=warmup_s,
        wanted=wanted,
        home=census_home(),
        built_once=built_once,
        req_s=req_s,
    )


def _warm_load_s(engine: Engine, run_dir: Path, load_s: float) -> tuple[float, bool]:
    """The measured start less what vLLM builds once and caches (see ``one_time_s``), so a
    census start, which runs on warm caches, is not judged by the cold one; and whether the
    start held such builds."""
    if engine.family != "vllm":
        return load_s, False
    try:
        log = (run_dir / "server.log").read_text("utf-8", errors="replace")
        built = start_phases.one_time_s(log)
    except (OSError, ValueError):
        return load_s, False
    return max(load_s - built, 0.0), built > 0


def expected_census_s(
    engine: Engine, runtime: Runtime, settings: Settings, load_s: float
) -> float | None:
    """Seconds a census start would add to this analysis, or None when none will run."""
    try:
        plan = _plan(engine, runtime, settings, None, load_s, 0.0, True)
    except Exception:
        return None
    return plan.expected_s if isinstance(plan, policy.CensusPlan) else None


def run_census(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    ready_timeout_s: float,
    *,
    version: str | None,
    req_s: float | None,
    gpu_model: str | None,
    wanted: bool = True,
) -> bool:
    """Plan the census for this run and run it, or record why it is skipped; whether Ctrl-C
    stopped it. Nothing in it can fail the analysis: a problem is the census record's reason."""
    try:
        return _run_census(
            project, engine, runtime, settings, run_dir, ready_timeout_s,
            version=version, req_s=req_s, gpu_model=gpu_model, wanted=wanted,
        )  # fmt: skip
    except Exception as error:
        said = str(error).partition("\n")[0] or type(error).__name__
        reason = f"the census could not be set up: {said}"
        with contextlib.suppress(Exception):
            write_record(run_dir, CensusRecord("failed", reason))
        return False


def _run_census(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    ready_timeout_s: float,
    *,
    version: str | None,
    req_s: float | None,
    gpu_model: str | None,
    wanted: bool,
) -> bool:
    context = analysis()
    load_s, built_once = _warm_load_s(engine, run_dir, context.load_s or 0.0)
    plan = _plan(
        engine,
        runtime,
        settings,
        version,
        load_s,
        context.warmup_s or 0.0,
        wanted,
        built_once,
        req_s,
    )
    if isinstance(plan, policy.CensusSkip):
        write_record(run_dir, CensusRecord("skipped", plan.reason, advice=plan.advice))
        return False
    try:
        add_census(
            project, engine, runtime, settings, run_dir, plan, ready_timeout_s,
            req_s=req_s or 0.0, gpu_model=gpu_model,
            gpu_index=int(runtime.gpus[0]) if runtime.gpus else 0, version=version,
        )  # fmt: skip
    except KeyboardInterrupt:
        return True
    return False
