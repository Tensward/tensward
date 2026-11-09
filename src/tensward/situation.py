"""Facts derived from other facts: the model's kind, attention and size, the device's memory,
and how full the KV cache ran. ``derived`` reads plain fact values, so a stored run gets them
without being measured again; a derived fact whose input is unknown is left out."""

from __future__ import annotations

from collections.abc import Mapping

from .fact_values import UNKNOWN
from .platforms import datasheet_bandwidth_gbs

KV_STATES = ("roomy", "filling", "full", "preempting")
KV_FULL = 0.95
KV_QUEUED = (0.05, 0.9)
KV_FILLING = 0.5
SIZE_BANDS = ("<2B", "2-10B", "10-40B", "40-100B", ">100B")
_SIZE_EDGES = (2.0, 10.0, 40.0, 100.0)


def _number(values: Mapping[str, object], name: str) -> float | None:
    value = values.get(name, UNKNOWN)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _flag(values: Mapping[str, object], name: str) -> bool | None:
    value = values.get(name, UNKNOWN)
    return value if isinstance(value, bool) else None


def _ratio(top: float | None, bottom: float | None) -> float | None:
    return top / bottom if top is not None and bottom else None


def kv_state(values: Mapping[str, object]) -> str | None:
    """How full the KV cache ran: ``preempting`` once a request was preempted, ``full`` at
    the peak usage threshold or with queueing near it, ``filling`` from half, else ``roomy``."""
    preemptions, peak = _number(values, "run.preemptions"), _number(values, "run.kv_usage_peak")
    if preemptions is None:
        return None
    if preemptions > 0:
        return "preempting"
    if peak is None:
        return None
    if peak >= KV_FULL:
        return "full"
    queue_floor, usage_floor = KV_QUEUED
    if peak >= usage_floor:
        queued = _number(values, "run.queue_share")
        if queued is None:
            return None
        if queued >= queue_floor:
            return "full"
    return "filling" if peak >= KV_FILLING else "roomy"


def _weights_share(values: Mapping[str, object]) -> float | None:
    memory = _number(values, "gpu.memory_gib")
    if values.get("engine.name") == "llamacpp":
        return _ratio(_number(values, "run.device_weight_bytes"), memory and memory * 2**30)
    weights = _number(values, "model.weight_gib")
    method, quant = values.get("settings.quantization"), values.get("model.quant")
    if method is UNKNOWN or (method == "fp8" and not isinstance(quant, str)):
        return None
    online_fp8 = method == "fp8" and quant == "none"
    return _ratio(weights and weights * (0.5 if online_fp8 else 1), memory)


def _api_servers(values: Mapping[str, object]) -> float | None:
    """Frontend processes: one when the setting is absent or None, unknown when it reads
    UNKNOWN or not above zero."""
    if values.get("settings.api_server_count") is None:
        return 1.0
    count = _number(values, "settings.api_server_count")
    return count if count else None


def _kind(values: Mapping[str, object]) -> str | None:
    moe, linear = _flag(values, "anatomy.moe"), _flag(values, "anatomy.linear_attention")
    if moe is None:
        return None
    if moe:
        return "moe"
    if linear is None:
        return None
    return "hybrid" if linear else "dense"


def _attention(values: Mapping[str, object]) -> str | None:
    mla, linear = _flag(values, "anatomy.mla"), _flag(values, "anatomy.linear_attention")
    if mla is None:
        return None
    if mla:
        return "mla"
    if linear is None:
        return None
    if linear:
        return "hybrid_linear"
    kv_heads, q_heads = _number(values, "anatomy.kv_heads"), _number(values, "anatomy.q_heads")
    if kv_heads is None or q_heads is None:
        return None
    return "mha" if kv_heads == q_heads else "gqa"


def _expert_share(values: Mapping[str, object]) -> float | None:
    used, experts = _number(values, "anatomy.experts_per_token"), _number(values, "anatomy.experts")
    return _ratio(used, experts)


def _active_params_b(values: Mapping[str, object]) -> float | None:
    params = _number(values, "model.params_b")
    if params is None:
        return None
    moe = _flag(values, "anatomy.moe")
    if moe is False:
        return params
    routed = _ratio(_number(values, "anatomy.routed_experts"), _number(values, "model.weight_gib"))
    active = _expert_share(values)
    if not moe or routed is None or active is None:
        return None
    return params * (1 - routed + routed * active)


def _size_band(active_params_b: float | None) -> str | None:
    if active_params_b is None:
        return None
    return SIZE_BANDS[sum(active_params_b >= edge for edge in _SIZE_EDGES)]


def _host_time_share(values: Mapping[str, object]) -> float | None:
    host_bytes, bandwidth = (
        _number(values, "run.host_bytes_per_token"),
        _number(values, "host.mem_bw_gbs"),
    )
    step_ms = _number(values, "run.tpot_p95_ms")
    if host_bytes is None or not bandwidth or not step_ms:
        return None
    return min(host_bytes / (bandwidth * 1e9) / (step_ms / 1e3), 1.0)


def _weights_on_host(values: Mapping[str, object]) -> float | None:
    host, device = (
        _number(values, "run.host_weight_bytes"),
        _number(values, "run.device_weight_bytes"),
    )
    return _ratio(host, None if host is None or device is None else host + device)


def _memory_domain(values: Mapping[str, object]) -> str | None:
    if values.get("platform.name") == "apple":
        return "unified"
    on_cpu = "cpu" in (values.get("run.compute"), values.get("engine.backend"))
    return "discrete" if isinstance(values.get("gpu.model"), str) and not on_cpu else None


def _bandwidth(values: Mapping[str, object]) -> float | None:
    model = values.get("gpu.model")
    return datasheet_bandwidth_gbs(model) if isinstance(model, str) else None


def derived(values: Mapping[str, object]) -> dict[str, object]:
    """The derived facts ``values`` has the inputs for. ``values`` maps fact names to values,
    UNKNOWN or absent for a fact not measured."""
    llamacpp = values.get("engine.name") == "llamacpp"
    active = _active_params_b(values)
    per_server = _ratio(_number(values, "run.req_s"), _api_servers(values))
    spread = _ratio(
        _number(values, "workload.prompt_tokens_p95"), _number(values, "workload.prompt_tokens_p50")
    )
    possible, hit = (
        _number(values, "workload.possible_prefix_hit"),
        _number(values, "run.prefix_hit_rate"),
    )
    found: dict[str, object | None] = {
        "run.kv_state": None if llamacpp else kv_state(values),
        "run.weights_share": _weights_share(values),
        "model.kind": _kind(values),
        "model.attention": _attention(values),
        "model.active_params_b": active,
        "model.size_band": _size_band(active),
        "model.active_expert_share": _expert_share(values),
        "gpu.mem_bw_gbs": _bandwidth(values),
        "host.memory_domain": _memory_domain(values),
        "run.prefix_headroom": None if possible is None or hit is None else possible - hit,
        "workload.prompt_spread": spread,
        "run.req_s_per_api_server": None if llamacpp else per_server,
        "run.host_time_share": _host_time_share(values) if llamacpp else None,
        "run.weights_on_host": _weights_on_host(values) if llamacpp else None,
        "run.clients_per_slot": _ratio(
            _number(values, "workload.clients"), _number(values, "run.slots")
        )
        if llamacpp
        else None,
    }
    return {name: value for name, value in found.items() if value is not None}
