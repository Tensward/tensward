"""The report's "Where the model lives" section and the advice a run's engine attaches to it."""

from __future__ import annotations

from typing import Any, Mapping

from ..engines.llamacpp.placement import WIRED_LIMIT, wired_limit_lines
from .model import Section, Text, item

GIB = 2**30
REGIMES = {
    "full_gpu": "all weights and the KV cache are in device memory",
    "dense_offload": "some layers run from host memory, the rest from device memory",
    "expert_offload": "the experts of some layers stay in host memory, the rest of the model is "
    "in device memory",
    "kv_host": "the weights are in device memory and the KV cache is in host memory",
    "cpu_only": "the model runs on the CPU, from host memory",
}
STEP_WORDS = {
    "host_expert_layers": "--n-cpu-moe",
    "host_layers": "--n-gpu-layers",
}


def _gib(size: float) -> str:
    return f"{size / GIB:.1f} GiB"


def _regime(block: Mapping[str, Any]) -> str:
    shape = block["regime"]
    text = f"regime: {REGIMES[shape['placement']]}"
    if shape["placement"] == "expert_offload":
        text += f" (--n-cpu-moe {block['placement']['host_expert_layers']})"
    if shape["host_memory_tight"]:
        text += "; host memory is tight, so weights may be paged from disk"
    if shape["numa_split"]:
        text += "; the host has several NUMA nodes, so reads can cross sockets"
    return text


def _held(device: int, host: int, *, cpu_only: bool) -> str:
    if cpu_only:
        return f"{_gib(device + host)} in host memory"
    return f"{_gib(device)} on the device, {_gib(host)} in host memory"


def _residence(block: Mapping[str, Any], facts: Mapping[str, Any]) -> list[str]:
    placed = block["placement"]
    cpu_only = block["regime"]["placement"] == "cpu_only"
    weights = _held(placed["device_weight_bytes"], placed["host_weight_bytes"], cpu_only=cpu_only)
    lines = [f"weights: {weights}"]
    kv_device, kv_host = facts.get("kv_device_bytes"), facts.get("kv_host_bytes")
    if kv_device is not None and kv_host is not None:
        lines.append(f"KV cache: {_held(kv_device, kv_host, cpu_only=cpu_only)}")
    return lines


def _host_line(model: Mapping[str, Any] | None, host: Mapping[str, Any] | None) -> str:
    host = host or {}
    launched = host.get("method") == "launch"
    if model and model.get("host_floor"):
        line = (
            f"host memory bandwidth: at least {model['bw_host_gbs']:.0f} GB/s (this step's "
            "decode time)"
        )
        if launched:
            line += f"; the probe launch measured {host['mem_bw_gbs']:.0f} GB/s"
        return line
    if not host:
        return ""
    if launched:
        return f"host memory bandwidth: {host['mem_bw_gbs']:.0f} GB/s (measured by a probe launch)"
    return f"host memory bandwidth: at least {host['mem_bw_gbs']:.0f} GB/s (this run's decode step)"


def _bandwidths(block: Mapping[str, Any], probes: Mapping[str, Any]) -> list[str]:
    model = block.get("decode_model")
    lines = []
    if line := _host_line(model, probes.get("host")):
        lines.append(line)
    else:
        why = probes.get("host_reason") or "no probe ran for this launch"
        lines.append(f"host memory bandwidth: not measured ({why})")
    if model:
        lines.append(f"device memory bandwidth: {model['bw_dev_gbs']:.0f} GB/s (datasheet)")
    pcie = probes.get("pcie")
    if pcie and block["placement"]["host_weight_bytes"]:
        if pcie.get("gbs_effective"):
            theory = pcie.get("theoretical_gbs")
            of = f" of {theory:.1f} GB/s theoretical" if theory else ""
            lines.append(f"PCIe: {pcie['gbs_effective']:.1f} GB/s effective{of}")
        elif pcie.get("stored_gbs"):
            lines.append(
                f"PCIe: {pcie['stored_gbs']:.1f} GB/s effective (measured by an earlier run on "
                f"this host; not measured on this run: {pcie.get('reason') or 'no slope'})"
            )
        else:
            lines.append(f"PCIe: not measured ({pcie.get('reason') or 'no slope was recorded'})")
    return lines


def _limit(block: Mapping[str, Any], probes: Mapping[str, Any]) -> str:
    placed, model = block["placement"], block.get("decode_model")
    host = probes.get("host") or {}
    if block["regime"]["placement"] == "cpu_only":
        if host.get("method") == "launch":
            host_ms = placed["host_bytes_per_token"] / (host["mem_bw_gbs"] * 1e6)
            step = block.get("step_ms")
            if step and 0 <= 1 - host_ms / step <= 0.3:
                return (
                    "limiting resource: host memory bandwidth (reading the weights takes "
                    f"{host_ms:.1f} ms of the {step:.1f} ms decode step)"
                )
            return (
                "limiting resource: not clear (the weight reads take "
                f"{host_ms:.1f} ms a token at the measured bandwidth, which does not explain "
                "the decode step)"
            )
        return "limiting resource: not clear (no probe launch measured the host memory bandwidth)"
    if model is None:
        return (
            "limiting resource: not clear (it needs both the host memory bandwidth and the "
            "device's, and one was not measured)"
        )
    step = block.get("step_ms")
    unexplained = block.get("overhead_share")
    if not (model["calibrated"] or model.get("host_floor")):
        streams = placed["host_weight_bytes"] / 1e9
        how = (
            f"{unexplained:.0%} of each decode step is time the weight reads do not explain"
            if unexplained is not None and unexplained >= 0
            else "the weight reads take longer than the decode step"
        )
        if block.get("topology") in ("unified", "host"):
            return (
                f"limiting resource: not clear, so host and device memory cannot be ranked: "
                f"{how}, at {model['concurrency']} requests at once"
            )
        return (
            f"limiting resource: not clear, so host and device memory cannot be ranked: {how}, "
            f"at {model['concurrency']} requests at once (reading a long prompt streams "
            f"{streams:.1f} GB over PCIe per batch)"
        )
    host_ms = placed["host_bytes_per_token"] / (model["bw_host_gbs"] * 1e6)
    dev_ms = placed["device_bytes_per_token"] / (model["bw_dev_gbs"] * 1e6)
    name = "host memory bandwidth" if host_ms >= dev_ms else "device memory bandwidth"
    of = f" of the {step:.1f} ms decode step" if step else ""
    at_most = "at most " if model.get("host_floor") else ""
    return (
        f"limiting resource: {name} (host reads take {at_most}{host_ms:.1f} ms{of}, device "
        f"reads {dev_ms:.1f} ms)"
    )


def _overhead(block: Mapping[str, Any], probes: Mapping[str, Any]) -> str | None:
    share, model = block.get("overhead_share"), block.get("decode_model")
    if share is None or share < 0 or (model or {}).get("host_floor"):
        return None
    return f"decode time the reads do not explain: {share:.0%} of the measured decode step"


def _unified(block: Mapping[str, Any], facts: Mapping[str, Any]) -> list[str]:
    """The one pool of a Mac and the two bounds on what a model may take from it."""
    if block.get("topology") != "unified":
        return []
    env = block.get("environment") or {}
    pool = f"one {env['mem_gib']:g} GiB pool" if env.get("mem_gib") else "one memory pool"
    working_set = facts.get("working_set_bytes")
    reserve = block.get("system_reserve_gib")
    parts = [pool]
    if working_set:
        parts.append(f"Metal working set {_gib(working_set)}")
    if reserve:
        parts.append(f"system memory keeps {reserve:.1f} GiB back")
    lines = ["; ".join(parts)]
    fit = block.get("fit") or {}
    if fit and (not fit["fits"] or fit.get("tight")):
        lines.append(f"fit: {fit['reason']}")
    if block.get("too_big"):
        lines.append(block["too_big"])
    return lines


def _same_machine(block: Mapping[str, Any]) -> list[str]:
    env = block.get("environment") or {}
    if env.get("os_family") != "macos":
        return []
    cores = env.get("logical_cpus")
    shared = f"its {cores} cores" if cores else "its cores"
    return [f"the load client ran on this Mac and shared {shared} with the server"]


def _fit_count(block: Mapping[str, Any], facts: Mapping[str, Any]) -> list[str]:
    """The line for a run whose own ``--fit`` placed layers on a Mac, where llama.cpp counts the
    working set and system memory as two pools."""
    if block.get("system_holds") is not False:
        return []
    if not {"n_gpu_layers", "tensor_overrides"} & set(facts.get("fit_changed") or ()):
        return []
    return [
        "llama.cpp's --fit counts the Metal working set and system memory separately; on this Mac "
        "they are the same memory, and the layers it placed need more than the system leaves free"
    ]


def _wired_limit(block: Mapping[str, Any]) -> list[str]:
    """The advice that raises the Metal working set: the command to run, how to undo it, and that
    Tensward leaves running it to the user."""
    step = next((s for s in block.get("next_steps") or [] if s["lever"] == WIRED_LIMIT), None)
    if step is None:
        return []
    return wired_limit_lines(step["value"], step["why"], block["environment"]["wired_limit_mb"])


def _steps(block: Mapping[str, Any]) -> list[str]:
    steps = block.get("next_steps") or []
    moves = [step for step in steps if step["lever"] != WIRED_LIMIT]
    if not steps:
        why = block.get("next_steps_cant_tell")
        return [f"next placement step: can't tell here ({why})"] if why else []
    lines = []
    for number, step in enumerate(moves):
        saved = step["saved_ms_per_token"]
        about = "about " if step["label"] == "rough estimate" else ""
        gain = (
            f"predicted saving {about}{saved:.{0 if about else 1}f} ms a token"
            if saved
            else "saving not predicted"
        )
        lead = "next placement step" if number == 0 else "then"
        lines.append(f"{lead}: {step['why']}; {gain} ({step['label']})")
    return lines


def placement_section(
    block: Mapping[str, Any] | None, probes: Mapping[str, Any] | None, facts: Mapping[str, Any]
) -> Section | None:
    """Where the weights and the KV cache sit, what limits decode and the placement step to try
    next; None when the engine has no placement model for the run."""
    if not block:
        return None
    probes = probes or {}
    if "unplaced" in block:
        line = f"where the model lives: can't tell ({block['unplaced']})"
        return Section(
            id="placement", title="Where the model lives", blocks=(item("placement", line),)
        )
    lines = [_regime(block), *_unified(block, facts), *_residence(block, facts)]
    lines += _same_machine(block)
    if block["regime"]["placement"] == "full_gpu":
        if block.get("topology") == "unified" and block.get("next_steps"):
            lines += _steps(block)
            lines += _wired_limit(block)
    else:
        lines += _bandwidths(block, probes)
        lines.append(_limit(block, probes))
        if overhead := _overhead(block, probes):
            lines.append(overhead)
        lines += _steps(block)
        lines += _wired_limit(block)
    lines += _fit_count(block, facts)
    return Section(
        id="placement",
        title="Where the model lives",
        blocks=tuple(item("placement", line) for line in lines),
    )


def advice_section(block: Mapping[str, Any] | None) -> Section | None:
    """The engine's advice lines for the run; a line with a short terminal text shows that on
    the terminal and the full text in report.md."""
    blocks: list[Text] = []
    for line in (block or {}).get("advice") or []:
        key = f"advice.{line['key']}"
        if short := line.get("terminal"):
            blocks.append(Text(key=key, text=line["text"], item=True, audience="markdown"))
            blocks.append(Text(key=key, text=short, item=True, audience="terminal"))
        else:
            blocks.append(Text(key=key, text=line["text"], item=True))
    return Section(id="advice", title="Advice", blocks=tuple(blocks))
