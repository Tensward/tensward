"""One plain line on where a run stands: how busy the GPU was, how much device and host memory
it used, what sits on the CPU, and what limits it. It reads only a run's recorded
``metrics.json``, so it is the same for every engine and platform; a signal the run did not
measure is named "not measured", never guessed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from .classify import power_cap_words
from .gpu_state import GpuState

GIB = 2**30
MIB_PER_GIB = 1024


def _power_cap(state: Mapping[str, Any]) -> str | None:
    names = {f.name for f in fields(GpuState)}
    gpu = GpuState(**{k: v for k, v in state.items() if k in names})
    if gpu.sm_mhz is None or not gpu.sm_max_mhz or not gpu.power_limit_w:
        return None
    return power_cap_words(gpu)


def _limit(metrics: Mapping[str, Any]) -> str:
    diagnosis = metrics.get("diagnosis") or {}
    primary = diagnosis.get("primary")
    if not diagnosis.get("requests", 1):
        return "limit not named, no request succeeded"
    if primary in (None, "load_limited") and (unexplained := diagnosis.get("unexplained")):
        return str(unexplained)
    if primary is None:
        return "no limit found"
    if primary == "offload":
        regime = (metrics.get("placement") or {}).get("regime") or {}
        what = "experts" if regime.get("placement") == "expert_offload" else "layers"
        return f"limited by reading the CPU-offloaded {what} each step"
    if primary == "power_cap" and (words := _power_cap(diagnosis.get("gpu_state") or {})):
        return f"limited by {words}"
    if primary == "load_limited":
        return "the server was not the limit at this load"
    name = (diagnosis.get("names") or {}).get(primary, primary)
    return f"limited by {name}"


def _gpu(metrics: Mapping[str, Any]) -> list[str]:
    state = (metrics.get("diagnosis") or {}).get("gpu_state") or {}
    parts = []
    busy = state.get("utilization")
    parts.append(
        f"GPU active {busy:.0f}% of the time" if busy is not None else "GPU activity not measured"
    )
    ceilings = metrics.get("ceilings") or {}
    if (pct := ceilings.get("decode_pct_of_ceiling")) is not None:
        parts.append(f"decode at {pct:.0f}% of its ceiling")
    peak, total = state.get("memory_peak_mib"), state.get("memory_total_mib")
    if peak is not None and total:
        parts.append(
            f"VRAM {peak / MIB_PER_GIB:.1f} of {total / MIB_PER_GIB:.1f} GiB used "
            f"({(total - peak) / MIB_PER_GIB:.1f} free)"
        )
    else:
        parts.append("VRAM not measured")
    return parts


def _host(metrics: Mapping[str, Any]) -> list[str]:
    block = metrics.get("placement") or {}
    env = block.get("environment") or {}
    total, free = env.get("mem_gib"), env.get("mem_available_gib")
    parts = []
    if total and free is not None:
        parts.append(f"host RAM {total - free:.1f} of {total:.1f} GiB used")
    else:
        parts.append("host RAM not measured")
    placed = block.get("placement") or {}
    experts, layers = placed.get("host_expert_layers") or 0, placed.get("host_layers") or 0
    held = (placed.get("host_weight_bytes") or 0) / GIB
    if experts > layers:
        parts.append(f"experts of {experts} layers on the CPU ({held:.1f} GiB)")
    elif layers:
        parts.append(f"{layers} layers on the CPU ({held:.1f} GiB)")
    return parts


def where_it_stands(metrics: Mapping[str, Any]) -> str:
    """The "Where it stands" line of a run, from its recorded metrics."""
    return f"Where it stands: {', '.join([*_gpu(metrics), *_host(metrics)])}; {_limit(metrics)}"


Row = tuple[str, str, str, str]


def _at(metrics: Mapping[str, Any] | None, *path: str) -> Any:
    found: Any = metrics
    for key in path:
        found = found.get(key) if isinstance(found, Mapping) else None
    return found


def number(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.3g}"


def _duration(ms: float) -> str:
    """A latency in milliseconds as "310 ms" or "2.65 s"."""
    return f"{number(ms / 1000)} s" if ms >= 1000 else f"{number(ms)} ms"


def _relative(was: float, now: float) -> str:
    if not was:
        return ""
    return "same" if abs(now - was) < 0.005 * abs(was) else f"{(now - was) / was:+.0%}"


def _points(was: float, now: float) -> str:
    return "same" if round(now - was) == 0 else f"{now - was:+.0f} pts"


def _gib(was: float, now: float) -> str:
    return "same" if abs(now - was) < 0.05 else f"{now - was:+.1f} GiB"


def _total(metrics: Mapping[str, Any] | None, *path: str) -> float | None:
    found = _at(metrics, *path)
    return found if isinstance(found, (int, float)) and found else None


def _on_cpu(metrics: Mapping[str, Any] | None) -> tuple[str, int] | None:
    """What sits on the CPU ("experts" or "layers") and of how many layers."""
    placed = _at(metrics, "placement", "placement") or {}
    experts, layers = placed.get("host_expert_layers") or 0, placed.get("host_layers") or 0
    if experts > layers:
        return "Experts", experts
    if layers:
        return "Layers", layers
    return None


def _host_used(metrics: Mapping[str, Any] | None) -> float | None:
    env = _at(metrics, "placement", "environment") or {}
    total, free = env.get("mem_gib"), env.get("mem_available_gib")
    return total - free if total and free is not None else None


SPEEDS: tuple[tuple[str, str, Any], ...] = (
    ("Requests/s", "request_throughput", number),
    ("Output tok/s", "output_throughput", number),
    ("First word, typical", "ttft_p50_ms", _duration),
    ("First word, slowest 1 in 20", "ttft_p95_ms", _duration),
    ("Per token, typical", "tpot_p50_ms", _duration),
    ("Per token, slowest 1 in 20", "tpot_p95_ms", _duration),
)


def stands_rows(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
    goal: tuple[str, float, float, float] | None = None,
    *,
    speeds: bool = True,
    significant: Mapping[str, bool] | None = None,
) -> list[Row]:
    """The speeds and where the run stands, before and after a change, as table rows of what,
    before, after and change; ``goal`` (a metrics key, before, after) is the figure the verdict
    judged, shown in place of that key's, with its change; ``speeds`` False leaves the speeds
    out. A speed's change shows only when ``significant`` (per metrics key) says it moved beyond
    its noise, "no change" when it did not, and nothing when its noise is not known. A speed
    neither run measured is left out; any other signal a run did not measure is "not
    measured"."""
    judged = significant or {}

    def speed_change(key: str, was: float, now: float) -> str:
        moved = _relative(was, now)
        if key not in judged or moved == "same":
            return moved if key in judged else ""
        if not judged[key]:
            return "no change"
        return f"{goal[3]:+.0%}" if goal is not None and goal[0] == key else moved

    rows: list[Row] = []

    def add(label: str, was: Any, now: Any, shown: Any, moved: Any, always: bool = True) -> None:
        if was is None and now is None and not (always and before is None):
            return
        change = moved(was, now) if was is not None and now is not None else ""
        rows.append(
            (
                label,
                "not measured" if was is None else shown(was),
                "not measured" if now is None else shown(now),
                change,
            )
        )

    for label, key, shown in SPEEDS if speeds else ():
        was, now = _at(before, key), _at(after, key)
        if goal is not None and goal[0] == key:
            was, now = goal[1], goal[2]
        add(label, was, now, shown, lambda a, b, k=key: speed_change(k, a, b), always=False)
    busy = ("diagnosis", "gpu_state", "utilization")
    add(
        "GPU active",
        _at(before, *busy),
        _at(after, *busy),
        lambda v: f"{v:.0f}% of the time",
        _points,
    )
    ceiling = ("ceilings", "decode_pct_of_ceiling")
    add(
        "Decode vs ceiling",
        _at(before, *ceiling),
        _at(after, *ceiling),
        lambda v: f"{v:.0f}%",
        _points,
        always=False,
    )
    peak = ("diagnosis", "gpu_state", "memory_peak_mib")
    total = _total(after, "diagnosis", "gpu_state", "memory_total_mib")
    label = f"VRAM used (of {total / MIB_PER_GIB:.1f} GiB)" if total else "VRAM used"
    was, now = (
        None if (v := _at(run, *peak)) is None else v / MIB_PER_GIB for run in (before, after)
    )
    add(label, was, now, lambda v: f"{v:.1f} GiB", _gib)
    host = _total(after, "placement", "environment", "mem_gib")
    label = f"Host RAM used (of {host:.1f} GiB)" if host else "Host RAM used"
    add(label, _host_used(before), _host_used(after), lambda v: f"{v:.1f} GiB", _gib)
    was, now = _on_cpu(before), _on_cpu(after)
    if was or now:
        what = (now or was or ("Layers", 0))[0]
        rows.append(
            (
                f"{what} on the CPU",
                f"{was[1]} layers" if was else "none",
                f"{now[1]} layers" if now else "none",
                "same" if was == now else "",
            )
        )
    if before is None:
        rows = [(label, now, "", "") for label, _, now, _ in rows]
    return rows


def limit_word(metrics: Mapping[str, Any]) -> str:
    """What limits a run, as a table value: "reading the CPU-offloaded experts each step"."""
    return _limit(metrics).removeprefix("limited by ")


def limit_line(before: Mapping[str, Any] | None, after: Mapping[str, Any]) -> str:
    """What limits the run, and what limited it before when that differs."""
    now = _limit(after)
    said = now[:1].upper() + now[1:]
    if before is None or (was := _limit(before)) == now:
        return f"{said}."
    return f"{said} (before: {was})."
