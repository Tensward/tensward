"""Reading census snapshots (schema 2.x) and taking the difference of two of them."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_MAJOR = 2
KNOWN = frozenset(
    {
        "census",
        "names_overflow",
        "kinds",
        "pid",
        "final",
        "t_first_ns",
        "t_last_ns",
        "busy_union_ns",
        "n_kernels",
        "n_records",
        "dropped",
        "late",
        "late_ns",
        "pending",
        "flushes",
        "paused",
        "api",
        "sync",
        "now_ns",
        "gaps",
        "copies",
        "syncs",
        "apis",
        "kernels",
        "lib",
        "graph_traces",
        "copies_detail",
        "memory",
    }
)


class SnapshotInvalid(ValueError):
    """The data is not a census snapshot this reader understands."""


@dataclass(frozen=True)
class Counted:
    n: int = 0
    ns: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class KernelRow:
    name: str
    n: int
    ns: int
    id: int = -1
    min_ns: int = 0
    max_ns: int = 0
    blocks: int = 0
    blocks_min: int = 0
    blocks_max: int = 0
    tpb: int = 0
    tpb_max: int = 0
    regs: int = 0
    shm_static: int = 0
    shm_dynamic_max: int = 0
    graph_n: int = 0
    queue_ns: int = 0


@dataclass(frozen=True)
class Snapshot:
    census: str
    pid: int
    n_kernels: int
    kernels: tuple[KernelRow, ...]
    names_overflow: int = 0
    kinds: tuple[str, ...] = ()
    final: bool = False
    t_first_ns: int = 0
    t_last_ns: int = 0
    busy_union_ns: int = 0
    n_records: int = 0
    dropped: int = 0
    late: int = 0
    late_ns: int = 0
    pending: int = 0
    flushes: int = 0
    paused: bool = False
    api: bool = False
    sync: bool = False
    now_ns: int = 0
    gaps: tuple[Counted, ...] = ()
    copies: Mapping[int, Counted] = field(default_factory=dict)
    syncs: Mapping[int, Counted] = field(default_factory=dict)
    apis: tuple[Mapping[str, Any], ...] = ()
    lib: Mapping[str, Any] | None = None
    graph_traces: Counted | None = None
    copies_detail: Mapping[str, Counted] = field(default_factory=dict)
    memory: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def minor(self) -> int:
        return int(self.census.split(".")[1]) if "." in self.census else 0


def _int(data: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotInvalid(f"{key} is not a number")
    return int(value)


def _counted(raw: Any) -> Counted:
    if not isinstance(raw, Mapping):
        raise SnapshotInvalid("a count is not an object")
    return Counted(_int(raw, "n"), _int(raw, "ns"), _int(raw, "bytes"))


def _kernel(raw: Any) -> KernelRow:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
        raise SnapshotInvalid("a kernel row has no name")
    return KernelRow(
        name=raw["name"],
        n=_int(raw, "n"),
        ns=_int(raw, "ns"),
        **{
            key: _int(raw, key, -1 if key == "id" else 0)
            for key in (
                "id",
                "min_ns",
                "max_ns",
                "blocks",
                "blocks_min",
                "blocks_max",
                "tpb",
                "tpb_max",
                "regs",
                "shm_static",
                "shm_dynamic_max",
                "graph_n",
                "queue_ns",
            )
        },
    )


def _kind_map(raw: Any) -> dict[int, Counted]:
    if not isinstance(raw, Mapping):
        raise SnapshotInvalid("copies and syncs are not objects")
    try:
        return {int(kind): _counted(value) for kind, value in raw.items()}
    except ValueError as exc:
        raise SnapshotInvalid("a copy or sync kind is not a number") from exc


def parse_snapshot(data: Any) -> Snapshot:
    try:
        return _parse(data)
    except (TypeError, AttributeError) as exc:
        raise SnapshotInvalid("a field has the wrong shape") from exc


def _parse(data: Any) -> Snapshot:
    if not isinstance(data, Mapping):
        raise SnapshotInvalid("a snapshot is a JSON object")
    version = data.get("census")
    if not isinstance(version, str):
        raise SnapshotInvalid("no census version")
    major, _, minor = version.partition(".")
    if major != str(SUPPORTED_MAJOR) or not minor.isdigit():
        raise SnapshotInvalid(f"census {version} is not a version 2 snapshot")
    for key in ("pid", "n_kernels", "kernels"):
        if key not in data:
            raise SnapshotInvalid(f"missing {key}")
    kernels = data["kernels"]
    if not isinstance(kernels, Sequence) or isinstance(kernels, str):
        raise SnapshotInvalid("kernels is not a list")
    gaps = data.get("gaps", ())
    apis = data.get("apis", ())
    if not isinstance(gaps, Sequence) or not isinstance(apis, Sequence):
        raise SnapshotInvalid("gaps and apis are lists")
    detail = data.get("copies_detail")
    traces = data.get("graph_traces")
    lib, memory = data.get("lib"), data.get("memory")
    if (lib is not None and not isinstance(lib, Mapping)) or (
        memory is not None and not isinstance(memory, Mapping)
    ):
        raise SnapshotInvalid("lib and memory are objects")
    return Snapshot(
        census=version,
        pid=_int(data, "pid"),
        n_kernels=_int(data, "n_kernels"),
        kernels=tuple(_kernel(row) for row in kernels),
        kinds=tuple(str(k) for k in data.get("kinds", ())),
        gaps=tuple(_counted(g) for g in gaps),
        copies=_kind_map(data.get("copies", {})),
        syncs=_kind_map(data.get("syncs", {})),
        apis=tuple(dict(a) for a in apis),
        lib=dict(lib) if lib is not None else None,
        memory=dict(memory) if memory is not None else None,
        graph_traces=_counted(traces) if traces is not None else None,
        copies_detail={str(k): _counted(v) for k, v in detail.items()}
        if isinstance(detail, Mapping)
        else {},
        extra={k: v for k, v in data.items() if k not in KNOWN},
        final=bool(data.get("final", False)),
        paused=bool(data.get("paused", False)),
        api=bool(data.get("api", False)),
        sync=bool(data.get("sync", False)),
        names_overflow=_int(data, "names_overflow"),
        t_first_ns=_int(data, "t_first_ns"),
        t_last_ns=_int(data, "t_last_ns"),
        busy_union_ns=_int(data, "busy_union_ns"),
        n_records=_int(data, "n_records"),
        dropped=_int(data, "dropped"),
        late=_int(data, "late"),
        late_ns=_int(data, "late_ns"),
        pending=_int(data, "pending"),
        flushes=_int(data, "flushes"),
        now_ns=_int(data, "now_ns"),
    )


def read_snapshots(out_dir: Path, prefix: str = "census") -> dict[int, Snapshot]:
    """Every readable ``<prefix>.<pid>.json`` in the directory, by pid; the rest are skipped."""
    found: dict[int, Snapshot] = {}
    for path in sorted(Path(out_dir).glob(f"{prefix}.*.json")):
        middle = path.name[len(prefix) + 1 : -len(".json")]
        if not middle.isdigit():
            continue
        try:
            found[int(middle)] = parse_snapshot(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return found


def read_timeline(path: Path, start_ns: int, end_ns: int) -> list[dict[str, int]]:
    """The timeline buckets in ``[start_ns, end_ns)``; a bucket written twice is added up."""
    buckets: dict[int, dict[str, int]] = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
            stamp = row["t_ns"]
            values = {
                k: v for k, v in row.items() if isinstance(v, int) and not isinstance(v, bool)
            }
        except (ValueError, KeyError, AttributeError, TypeError):
            continue
        if not isinstance(stamp, int) or not start_ns <= stamp < end_ns:
            continue
        seen = buckets.setdefault(stamp, {"t_ns": stamp})
        for key, value in values.items():
            if key != "t_ns":
                seen[key] = seen.get(key, 0) + value
    return [buckets[stamp] for stamp in sorted(buckets)]


@dataclass(frozen=True)
class CensusWindow:
    pids: tuple[int, ...]
    kernels: tuple[KernelRow, ...]
    copies: Mapping[int, Counted]
    syncs: Mapping[int, Counted]
    n_kernels: int
    dropped: int
    requests: int
    start_ns: int
    end_ns: int
    timeline: tuple[Mapping[str, int], ...]
    after: Mapping[int, Snapshot]
    gaps: tuple[Counted, ...] = ()
    copies_detail: Mapping[str, Counted] = field(default_factory=dict)
    graph_traces: Counted | None = None
    failed: int = 0
    scoped: bool = True
    out_tokens: int = 0  # output tokens of the window's completed requests


def _kernel_delta(now: KernelRow, then: KernelRow | None) -> KernelRow:
    if then is None:
        return now
    return KernelRow(
        name=now.name,
        n=now.n - then.n,
        ns=now.ns - then.ns,
        id=now.id,
        min_ns=now.min_ns,
        max_ns=now.max_ns,
        blocks=now.blocks - then.blocks,
        blocks_min=now.blocks_min,
        blocks_max=now.blocks_max,
        tpb=now.tpb,
        tpb_max=now.tpb_max,
        regs=now.regs,
        shm_static=now.shm_static,
        shm_dynamic_max=now.shm_dynamic_max,
        graph_n=now.graph_n - then.graph_n,
        queue_ns=now.queue_ns - then.queue_ns,
    )


def _minus(now: Counted, then: Counted | None) -> Counted:
    o = then or Counted()
    return Counted(now.n - o.n, now.ns - o.ns, now.bytes - o.bytes)


def _add(a: Counted, b: Counted) -> Counted:
    return Counted(a.n + b.n, a.ns + b.ns, a.bytes + b.bytes)


def _plus(
    total: dict[Any, Counted], now: Mapping[Any, Counted], then: Mapping[Any, Counted]
) -> None:
    for kind, c in now.items():
        total[kind] = _add(total.get(kind, Counted()), _minus(c, then.get(kind)))


def window(
    before: Mapping[int, Snapshot],
    after: Mapping[int, Snapshot],
    *,
    requests: int,
    start_ns: int,
    end_ns: int,
    timelines: Mapping[int, Sequence[Mapping[str, int]]],
    failed: int = 0,
    out_tokens: int = 0,
) -> CensusWindow:
    """What was recorded between the two snapshots, by the processes whose kernels grew."""
    empty = Snapshot(census="2.0", pid=0, n_kernels=0, kernels=())
    pids = tuple(
        sorted(p for p, s in after.items() if s.n_kernels > before.get(p, empty).n_kernels)
    )
    kernels: dict[str, KernelRow] = {}
    copies: dict[int, Counted] = {}
    syncs: dict[int, Counted] = {}
    detail: dict[str, Counted] = {}
    gaps: list[Counted] = []
    traces: Counted | None = None
    dropped = n_kernels = 0
    for pid in pids:
        now, then = after[pid], before.get(pid, empty)
        n_kernels += now.n_kernels - then.n_kernels
        dropped += now.dropped - then.dropped
        names = {k.name: k for k in then.kernels}
        for row in now.kernels:
            delta = _kernel_delta(row, names.get(row.name))
            if delta.n <= 0:
                continue
            held = kernels.get(delta.name)
            kernels[delta.name] = (
                delta
                if held is None
                else KernelRow(
                    **{
                        **held.__dict__,
                        "n": held.n + delta.n,
                        "ns": held.ns + delta.ns,
                        "blocks": held.blocks + delta.blocks,
                        "graph_n": held.graph_n + delta.graph_n,
                        "queue_ns": held.queue_ns + delta.queue_ns,
                    }
                )
            )
        _plus(copies, now.copies, then.copies)
        _plus(syncs, now.syncs, then.syncs)
        _plus(detail, now.copies_detail, then.copies_detail)
        for i, gap in enumerate(now.gaps):
            grown = _minus(gap, then.gaps[i] if i < len(then.gaps) else None)
            if i < len(gaps):
                gaps[i] = _add(gaps[i], grown)
            else:
                gaps.append(grown)
        if now.graph_traces is not None:
            grown = _minus(now.graph_traces, then.graph_traces)
            traces = grown if traces is None else _add(traces, grown)
    timeline = tuple(timelines.get(pids[0], ())) if len(pids) == 1 else ()
    return CensusWindow(
        pids=pids,
        kernels=tuple(sorted(kernels.values(), key=lambda k: -k.ns)),
        copies=copies,
        syncs=syncs,
        n_kernels=n_kernels,
        dropped=dropped,
        requests=requests,
        start_ns=start_ns,
        end_ns=end_ns,
        timeline=timeline,
        after={p: after[p] for p in pids},
        gaps=tuple(gaps),
        copies_detail=detail,
        graph_traces=traces,
        failed=failed,
        scoped=all(p in before for p in pids),
        out_tokens=out_tokens,
    )
