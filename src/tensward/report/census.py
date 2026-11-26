"""The GPU census in a report: a section of plain rows and the one line the terminal shows
first."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from ..census import inefficiency
from ..census.inefficiency import Inefficiency
from ..census.policy import INSTALL, INSTALL_END
from ..census.record import CensusRecord
from .model import Block, Section, Text, item

UNVERIFIED = "This engine build is not yet verified with the census."
BYTE_UNITS = ("bytes", "KiB", "MiB", "GiB")


def _bytes(value: float) -> str:
    size, unit = float(value), 0
    while size >= 1024 and unit < len(BYTE_UNITS) - 1:
        size, unit = size / 1024, unit + 1
    return f"{size:.0f} {BYTE_UNITS[unit]}" if unit == 0 else f"{size:.1f} {BYTE_UNITS[unit]}"


def _copy_row(where: str) -> object:
    return lambda x: f"{_bytes(x)} copied {where} per request"


ROWS = {
    "census.busy_share": lambda x: f"the GPU ran kernels {x:.0%} of the time",
    "census.kernel_ms_per_request": lambda x: (
        f"{x:.1f} ms of GPU kernels per request (recorded with CUDA graphs off)"
    ),
    "census.kernels_per_request": lambda x: f"{x:.0f} kernels launched per request",
    "census.copies_per_request": lambda x: f"{x:.1f} copies between host and GPU per request",
    "census.h2d_bytes_per_request": _copy_row("from host to GPU"),
    "census.d2h_bytes_per_request": _copy_row("from GPU to host"),
    "census.d2d_bytes_per_request": _copy_row("inside the GPU"),
    "census.fullness": lambda x: f"the GPU's warp slots were {x:.0%} full while it ran",
    "census.fullness_at_most": lambda x: (
        f"at most {x:.0%} full (an upper bound for vLLM on this card)"
    ),
    "census.graphs_off": lambda x: (
        "CUDA graphs were off in the census start"
        if x
        else "CUDA graphs were on in the census start"
    ),
    "census.recording_processes": lambda x: f"{x} process(es) recorded kernels",
}


def _origin(source: str | None) -> str:
    return f"This census is from run {source}, not from this run." if source else ""


def census_section(
    record: CensusRecord, offered: Collection[str] = (), source: str | None = None
) -> Section:
    """One plain row per reading with its meaning, or why there is no reading and what to do."""
    blocks: list[Block] = []
    if source:
        blocks.append(Text(key="census_origin", text=_origin(source)))
    if record.status != "ok":
        blocks.append(Text(key="census_reason", text=f"{_sentence(record.reason)}", item=True))
        if record.advice:
            blocks.append(Text(key="census_advice", text=record.advice))
    else:
        for name, value in record.readings.items():
            if name in ROWS:
                said = ROWS[name](value)  # type: ignore[operator]
                said += " (this points to a change below)" if name in offered else ""
                blocks.append(item("census_reading", said))
        blocks.append(
            Text(
                key="census_note",
                text=f"A separate diagnostic start of {record.wall_s:.0f} s, not a speed claim.",
            )
        )
    if _unverified(record):
        blocks.append(Text(key="census_unverified", text=UNVERIFIED))
    return Section(id="census", title="GPU census", blocks=tuple(blocks))


def _sentence(reason: str) -> str:
    return reason[:1].upper() + reason[1:] + ("" if reason.endswith(".") else ".") if reason else ""


def _unverified(record: CensusRecord) -> bool:
    return record.status != "skipped" and (record.library or {}).get("verified") is False


def census_line(
    record: CensusRecord,
    offered: Sequence[str] = (),
    source: str | None = None,
    covers: Sequence[str] = (),
) -> list[Text]:
    """The first-screen lines of the census; the advice sits above the "Next:" menu. ``offered``
    names the census offers this report makes, ``covers`` every census offer the engine has."""
    wall = f"diagnostic start, {record.wall_s:.0f} s"
    if record.status == "ok":
        readings = record.readings
        seen = []
        if "census.busy_share" in readings:
            seen.append(f"busy {readings['census.busy_share']:.0%}")
        if "census.kernel_ms_per_request" in readings:
            seen.append(f"{readings['census.kernel_ms_per_request']:.1f} ms of kernels per request")
        for key, label in (
            ("census.h2d_bytes_per_request", "H2D"),
            ("census.d2h_bytes_per_request", "D2H"),
        ):
            if key in readings:
                seen.append(f"{readings[key] / 1e9:.1f} GB {label} per request")
        head = f"GPU census: {', '.join(seen) or 'recorded'} ({wall})"
        if offered:
            text = f"{head}; it points to {', '.join(offered)} below (not a speed claim)."
        elif covers:
            text = (
                f"{head}. None of Tensward's census offers applies here (it checks: "
                f"{', '.join(covers)})."
            )
        else:
            text = f"{head}. Tensward has no census offer for this engine."
    else:
        text, advice = _not_run_line(record, covers)
    lines = [Text(key="census_line", text=text, audience="terminal")]
    if record.status == "ok":
        advice = record.advice or ""
    for extra in (_origin(source), UNVERIFIED if _unverified(record) else "", advice):
        if extra:
            lines.append(Text(key="census_note", text=extra, audience="terminal"))
    return lines


def _not_run_line(record: CensusRecord, covers: Sequence[str]) -> tuple[str, str]:
    """The line for a census that did not record, naming the offers it holds back and the
    install command when one is known, and the advice left for the line below."""
    reason, advice = record.reason.rstrip("."), record.advice or ""
    head = (
        f"GPU census skipped: {reason}"
        if record.status == "skipped"
        else f"GPU census: {reason}; the analysis is complete without it"
    )
    if covers:
        head += f"; offers that need it are held back ({', '.join(covers)})"
    if advice.startswith(INSTALL) and INSTALL_END in advice:
        command, advice = advice.removeprefix(INSTALL).split(INSTALL_END, 1)
        head += f"; install with: {command}"
    return head + ".", advice


CHECKED = {"launch_gaps": "launch gaps", "host_wait": "waiting on the host"}
HOST_WAIT = (
    "Up to {share:.0%} of the time the GPU waited on the host; a host-side change can give back "
    "at most this share."
)
PAGEABLE_WORK = (
    "copying from pinned host buffers instead of pageable or file-mapped memory, overlapping "
    "copies with compute, or a memory layout that keeps that data on the GPU"
)
COPY_WORK = (
    "overlapping copies with compute, moving fewer bytes per request, or a memory layout that "
    "keeps that data on the GPU"
)
HOST_WORK = (
    "overlapping that host work with the GPU (saving state asynchronously, pre-faulted or "
    "huge-page host buffers, an allocator that reuses large buffers), or moving it off the "
    "engine's main thread"
)
EVERY_WORK = (
    "copying from pinned host buffers instead of pageable or file-mapped memory, overlapping "
    "copies with compute, moving fewer bytes per request, a memory layout that keeps that data "
    "on the GPU, overlapping host work with the GPU (saving state asynchronously, pre-faulted "
    "or huge-page host buffers, an allocator that reuses large buffers), or moving host work "
    "off the engine's main thread"
)


def _launch_line(found: Inefficiency, engine: str) -> str:
    if engine == "llamacpp":
        return (
            f"Launch gaps: {found.share:.0%} of the time in gaps under 0.1 ms (recorded with "
            "CUDA graphs off; a setup with graphs on has fewer). Work that removes this: fusing "
            "small kernels into fewer, larger ones."
        )
    each = found.detail.get("kernel_us")
    average = f" ({each:.0f} µs per kernel on average)" if each is not None else ""
    return (
        f"Launch gaps: the GPU sat idle {found.share:.0%} of the time in gaps under 0.1 ms while "
        f"the next kernel was launched{average}. Work that removes this: capturing the engine "
        "step as a CUDA graph, or fusing small kernels into fewer, larger ones."
    )


def _host_wait_line(found: Inefficiency) -> str:
    detail = found.detail
    copies, host = detail.get("copy_part", 0.0), detail.get("host_part", 0.0)
    gb = detail.get("gb_per_request")
    per = f"{gb:.1f} GB per request" if gb is not None else None
    head = HOST_WAIT.format(share=found.share)
    label = found.label if inefficiency.LABEL_CONFIRMED else None
    if label == "pageable_copies" and per is not None:
        tail = (
            f"{copies:.0%} on copies between host and GPU ({per}, "
            f"{detail['pageable_h2d']:.0%} of host-to-GPU bytes from pageable memory). "
            f"Work that removes this: {PAGEABLE_WORK}."
        )
    elif label == "host_gpu_copies" and per is not None:
        tail = (
            f"{copies:.0%} on copies between host and GPU ({per}, "
            f"{detail['pinned_h2d']:.0%} of host-to-GPU bytes from pinned memory). "
            f"Work that removes this: {COPY_WORK}."
        )
    elif label == "host_work" and per is not None:
        tail = (
            f"The engine worked on the host between steps ({copies:.0%} on copies, {per}). "
            f"Work that removes this: {HOST_WORK}."
        )
    else:
        moved = f" ({per})" if per is not None else ""
        tail = (
            f"{copies:.0%} on copies between host and GPU{moved}, {host:.0%} on host work "
            f"between steps. Work that removes this: {EVERY_WORK}."
        )
    return f"{head} {tail[:1].upper()}{tail[1:]}"


def inefficiency_section(
    record: CensusRecord, engine: str, census_run: str | None = None
) -> Section | None:
    """What the settings can't fix: one line per confirmed class at or above its floor, or the
    classes checked when none is; None when no class is confirmed for the run's situation."""
    if record.status != "ok" or not record.inefficiency:
        return None
    confirmed = [found for found in record.inefficiency if found.kind in inefficiency.CONFIRMED]
    if not confirmed:
        return None
    blocks: list[Block] = []
    if census_run:
        blocks.append(
            Text(key="inefficiency_origin", text=f"Recorded on the setup of run {census_run}.")
        )
    large = [found for found in confirmed if found.share >= found.floor]
    for found in large:
        line = (
            _launch_line(found, engine) if found.kind == "launch_gaps" else _host_wait_line(found)
        )
        blocks.append(item(f"inefficiency_{found.kind}", line))
    if not large:
        kinds = ", ".join(CHECKED[found.kind] for found in confirmed)
        busy = record.readings.get("census.busy_share")
        ran = f": the GPU ran kernels {busy:.0%} of the time" if busy is not None else ""
        blocks.append(
            Text(
                key="inefficiency_none",
                text=f"None of the checked kinds is large here ({kinds}){ran}.",
            )
        )
    return Section(id="inefficiency", title="What the settings can't fix", blocks=tuple(blocks))
