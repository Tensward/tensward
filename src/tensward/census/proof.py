"""Whether the census really recorded the engine: the library and the engine's CUPTI in the
memory of the process that ran the kernels, the library's own line, a readable snapshot with
kernels in it, the right build and CUDA graphs off where the start turned them off."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from ..runtime import CensusLaunch, RunningServer, process_maps
from .snapshot import Snapshot

ON_LINE = re.compile(
    r"twcensus: on, pid (?P<pid>\d+), out (?P<out>\S+?)(?P<paused> \(paused\))?, "
    r"flush (?P<flush>\w+), census (?P<census>2\.\d+)\b.*"
)
OFF_LINE = re.compile(r"twcensus: off: (?P<reason>.+)")
NO_OUTPUT = "the census wrote nothing; its readings are not used"
CUPTI_FILE = re.compile(r"(/\S*/libcupti\.so[.\d]*)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class AttachProof:
    library_mapped: bool
    cupti_mapped: bool
    on_line: str | None
    off_reason: str | None
    snapshot_ok: bool
    n_kernels: int
    kind_ok: bool | None  # None: the snapshot does not say which build it is
    graphs_off_ok: bool
    pids: tuple[int, ...]
    kind: str | None = None
    expected_kind: str | None = None
    cupti_seen: str | None = None

    def attached(self) -> bool:
        """The library ran in the engine and said so."""
        return (
            self.off_reason is None
            and self.library_mapped
            and self.on_line is not None
            and self.cupti_mapped
            and self.kind_ok is not False
        )

    def passed(self) -> bool:
        return self.attached() and self.snapshot_ok and self.n_kernels > 0 and self.graphs_off_ok

    def reason(self) -> str:
        """Why the census readings cannot be used; empty when they can."""
        if self.off_reason is not None:
            return f"the census turned itself off: {self.off_reason}"
        if not self.library_mapped:
            return "the census library was not loaded into the engine"
        if self.on_line is None:
            return "the census library did not say it was recording"
        if not self.cupti_mapped:
            seen = f" ({self.cupti_seen})" if self.cupti_seen else ""
            return f"the engine did not load the CUPTI the census was started with{seen}"
        if self.kind_ok is False:
            return (
                f"the census library is the wrong kind: {self.kind}, expected {self.expected_kind}"
            )
        if not self.snapshot_ok:
            return NO_OUTPUT
        if self.n_kernels <= 0:
            return "the census recorded no kernels"
        if not self.graphs_off_ok:
            return "CUDA graphs were on in the census start"
        return ""


def _on_lines(log: str) -> dict[int, str]:
    found: dict[int, str] = {}
    for line in log.splitlines():
        start = line.find("twcensus: on,")
        matched = ON_LINE.match(line[start:]) if start >= 0 else None
        if matched and matched["flush"] == "ondemand":
            found.setdefault(int(matched["pid"]), matched[0])
    return found


def _off_reason(log: str) -> str | None:
    for line in log.splitlines():
        start = line.find("twcensus: off:")
        if start >= 0 and (matched := OFF_LINE.match(line[start:])):
            return matched["reason"].strip()
    return None


def _same_folder(mapped: str, folder: str, inside: bool) -> bool:
    if inside:
        return os.path.normpath(os.path.dirname(mapped)) == os.path.normpath(folder)
    return os.path.realpath(os.path.dirname(mapped)) == os.path.realpath(folder)


def prove_attach(
    server: RunningServer,
    launch: CensusLaunch,
    after: Mapping[int, Snapshot],
    *,
    expected_kind: str,
    graphs_off: bool,
) -> AttachProof:
    """Check the census of a running census start against ``after``, the snapshots read at the
    after moment. The process checked is the one that recorded kernels (vLLM's engine core, not
    its API server); without one, every process that wrote a snapshot, else the server."""
    log = server.log_text()
    recording = tuple(sorted(pid for pid, snap in after.items() if snap.n_kernels > 0))
    own = server.identity().get("pid")
    checked = recording or tuple(sorted(after)) or ((own,) if isinstance(own, int) else ())
    inside = server.identity().get("runtime") == "docker"
    library_mapped = cupti_mapped = bool(checked)
    seen: str | None = None
    for pid in checked:
        maps = process_maps(server, pid) or ""
        library_mapped = library_mapped and f"/{launch.library.name}" in maps
        mapped = CUPTI_FILE.findall(maps)
        seen = seen or (mapped[0] if mapped else None)
        cupti_mapped = cupti_mapped and any(
            _same_folder(path, launch.cupti_dir, inside) for path in mapped
        )
    lines = _on_lines(log)
    on_line = next((lines[pid] for pid in checked if pid in lines), None)
    if on_line is None and lines and not recording:
        on_line = next(iter(lines.values()))
    kinds = {str(lib["kind"]) for pid in recording if (lib := after[pid].lib) and lib.get("kind")}
    kind = ", ".join(sorted(kinds)) or None
    graphs_used = any(row.graph_n > 0 for pid in recording for row in after[pid].kernels)
    return AttachProof(
        library_mapped=library_mapped,
        cupti_mapped=cupti_mapped,
        on_line=on_line,
        off_reason=_off_reason(log),
        snapshot_ok=bool(after),
        n_kernels=sum(after[pid].n_kernels for pid in recording),
        kind_ok=None if not kinds else kinds == {expected_kind},
        graphs_off_ok=not (graphs_off and graphs_used),
        pids=recording,
        kind=kind,
        expected_kind=expected_kind,
        cupti_seen=None if cupti_mapped else seen,
    )
