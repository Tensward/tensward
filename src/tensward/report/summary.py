"""The plain-words summary a report opens with, and the one-line result its terminal ends with."""

from __future__ import annotations

import math

from ..classify import NAMES, Bottleneck
from ..playbook import GOAL_WORDS

LIMITS: dict[Bottleneck, str] = {
    "load_limited": "nothing in the server; the test did not push it to its limit, so it "
    "has room for more traffic",
    "queueing": "requests waiting in line before the server starts them",
    "kv_capacity": "memory for the conversations the server keeps (the KV cache) is full",
    "prefill": "reading the prompts (prefill) takes most of the work",
    "step_budget": "the server's limit on how many tokens it handles per step",
    "prefill_stalls_decode": "reading new prompts holds up the answers already streaming",
    "decode_bandwidth": "how fast the GPU can read the model's weights while writing answers",
    "gpu_compute": "the GPU's calculation speed",
    "gpu_saturated": "the GPU is fully busy",
    "host_overhead": "work on the CPU between GPU steps",
    "frontend_cpu": "the CPU work of the API server",
    "offload": "part of the model sits in CPU memory and is read over a slow link",
    "long_context": "very long prompts slow the attention step",
    "speculation": "speculative decoding that costs more than it saves",
    "multi_gpu": "traffic between GPUs",
}
GLOSSARY = (
    "TTFT is the time to the first token of an answer, TPOT the time per token after it; "
    "p95 is the slowest 1 in 20 requests."
)
FEW_TOOL_CALLS = 0.1  # of the requests that offered tools, below which tool use is "rare"
LOWERS = {
    "throughput": "raises requests per second",
    "ttft": "lowers time to first token",
    "tpot": "lowers time per token",
}
SIDE_GAINS = {
    "throughput": "more requests per second",
    "ttft": "a shorter time to first token",
    "tpot": "a lower time per token",
}


def health_line(succeeded: int, failed: int, req_s: float | None) -> str:
    total = succeeded + failed
    rate = f", {req_s:.2f} requests per second" if req_s else ""
    if total == 0 or succeeded == 0:
        return "Is the server healthy: no. No request succeeded; this run's server log says why."
    if failed == 0:
        return f"Is the server healthy: yes. All {total} requests were answered{rate}."
    return (
        f"Is the server healthy: partly. {failed} of {total} requests failed{rate}; "
        "see Requests below."
    )


def limit_line(
    primary: Bottleneck | None,
    judged: bool,
    name: str | None,
    host_share: int | None = None,
    unexplained: str | None = None,
) -> str:
    """What limits the run; ``host_share`` is the percent of each step spent reading the model
    from CPU memory when that was measured and is high, which beats "nothing limits it", and so
    does ``unexplained``, decode well below its ceiling with no limit named."""
    if not judged:
        return "What limits it: this run did not measure enough to say."
    if host_share is not None and primary in (None, "load_limited"):
        return (
            f"What limits it: part of the model is read from CPU memory, {host_share}% of each "
            "step."
        )
    if unexplained is not None and primary in (None, "load_limited"):
        return f"What limits it: {unexplained}."
    if primary is None:
        return "What limits it: no single limit stood out in this run."
    return f"What limits it: {LIMITS.get(primary) or name or NAMES[primary]}."


def nothing_found(goal: str) -> str:
    """ "nothing found that lowers time per token here"."""
    return f"nothing found that {LOWERS[goal]} here"


def side_heading(goal: str) -> str:
    """The line above the changes offered for another goal than ``goal``."""
    found = nothing_found(goal)
    return f"{found[:1].upper()}{found[1:]}. These may help elsewhere without making it worse:"


def side_label(side: str, goal: str) -> str:
    """ "more requests per second (TPOT is checked so it doesn't get worse)"."""
    words = GOAL_WORDS.get(goal, goal)
    return f"{SIDE_GAINS[side]} ({words} is checked so it doesn't get worse)"


def reveal(in_flight: float | None) -> str:
    """What would show a change to test when none applies: the load the test kept, when it did
    not push the server, else the test itself."""
    if in_flight is not None:
        return (
            f"your test kept only {in_flight:.0f} requests in flight; if real traffic is higher, "
            "run `tensward init` again with that concurrency"
        )
    return (
        "if your real prompts or traffic differ from this test, run `tensward init` again with them"
    )


def next_line(
    change: str | None,
    goal: str,
    seconds: float | None = None,
    *,
    side: str | None = None,
    hint: str | None = None,
) -> str:
    """The summary's one pointer to the Next menu: the same change and minutes it names; for a
    change offered for another goal (``side``), that goal; with no change, ``hint`` (``reveal``)."""
    if change is None:
        return f"Next: {nothing_found(goal)}; {hint or reveal(None)}."
    minutes = f" (~{math.ceil(seconds / 60)} min)" if seconds else ""
    if side is not None:
        return (
            f"Next: `tensward try` tests {change} for {SIDE_GAINS[side]}{minutes}; "
            f"{nothing_found(goal)}."
        )
    return f"Next: `tensward try` tests {change}{minutes}."


def tool_line(offered: int, calling: int, valid: float | None) -> str | None:
    """Both tool numbers together: how many requests called a tool, and how many calls were
    valid. None when there was no call to judge or the few-calls warning says it already."""
    if calling == 0 or tool_warning(offered, calling) is not None:
        return None
    return f"Tools: {tool_phrase(offered, calling, valid)}."


def tool_phrase(offered: int, calling: int, valid: float | None) -> str:
    """ "82 of 160 requests called a tool; all calls valid": the report's and try's wording."""
    how = "all calls valid" if valid is None or valid >= 1 else f"{valid:.0%} of calls valid"
    return f"{calling} of {offered} requests called a tool; {how}"


def tool_warning(offered: int, calling: int) -> str | None:
    """The sentence for a run that offered tools and rarely got a call: tool-call quality
    cannot be judged from it."""
    if offered == 0 or calling >= max(offered * FEW_TOOL_CALLS, 5):
        return None
    some = f"None of the {offered}" if calling == 0 else f"Only {calling} of {offered}"
    return (
        f"{some} requests called a tool, so tool-call quality cannot be checked; your "
        "real traffic may differ."
    )


def result_line(
    succeeded: int,
    failed: int,
    changes: int,
    goal: str,
    *,
    side: bool = False,
    hint: str | None = None,
) -> str:
    """One line for the end of the terminal: the server's state and what is left to test; the
    ``changes`` are for other goals than ``goal`` when ``side``; with none, ``hint``
    (``reveal``) says what would show one."""
    if succeeded == 0:
        return "Result: no request succeeded; read the log named above."
    state = (
        "your server answered every request"
        if failed == 0
        else (f"your server failed {failed} of {succeeded + failed} requests")
    )
    if changes == 0:
        said = hint or reveal(None)
        return f"Result: {state}; {nothing_found(goal)}. {said[:1].upper()}{said[1:]}."
    many = "1 change" if changes == 1 else f"{changes} changes"
    if side:
        return (
            f"Result: {state}; {nothing_found(goal)}, but {many} listed above may help "
            "elsewhere without making it worse."
        )
    return f"Result: {state}; {many} could still help."
