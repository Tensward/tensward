from tensward.cli_try import _not_better, prefill_gain
from tensward.rejected import PREFILL_GAIN, your_call


def _step(prompt_change=-0.42, change=0.019, broken=0):
    """GLM-5.3-Flash Q2's -ub 2048 try (GLM2 try 2): prompts 263 -> 152 s, decode 303 -> 337 s,
    requests per second inside the 16.8% band."""
    return {
        "metric": "req_s", "change": change, "band": 0.1675, "recommended": False,
        "verdict": "within_noise", "reason": None, "loss": None, "tool_calls": None,
        "answers": {"total": 20, "broken": broken, "differ": 19, "identical": 1},
        "phases": {
            "prompt": {"before_s": 263.0, "after_s": 263.0 * (1 + prompt_change),
                       "change": prompt_change, "band": 0.08,
                       "before_tok_s": 1180.0, "after_tok_s": 2035.0},
            "decode": {"before_s": 303.0, "after_s": 337.0, "change": 0.112, "band": 0.05},
        },
    }  # fmt: skip


def test_faster_prompts_with_the_goal_inside_its_band_are_offered_to_keep():
    step = _step()
    assert prefill_gain(step)
    text = _not_better(step)
    assert "prompts x1.72 faster" in text and "1,180 to 2,035 prompt tok/s" in text
    assert "decode time +11% (past its 5% band)" in text and your_call(text)
    assert PREFILL_GAIN in text
    assert text.startswith("Requests/s: paired +2%, inside its 17% band; prompts")
    assert "Nothing changed" not in text
    assert not prefill_gain(_step(prompt_change=-0.05))  # inside its band
    assert not prefill_gain(_step(change=-0.2))  # the goal got worse past its band
    assert not prefill_gain(_step(broken=1))


def test_a_request_rate_gain_is_put_down_to_shorter_replies_only_when_they_carry_it():
    from tensward.history import Compared, shorter_replies
    from tensward.window import Delta

    def found(rate, rate_band, tokens, tokens_band, verdict="within_noise"):
        return Compared("0001", {"req_s": Delta("req_s", rate, rate_band, "better", True),
                                 "output_tok_s": Delta("output_tok_s", tokens, tokens_band,
                                                       verdict, True)})  # fmt: skip

    ngram = shorter_replies(found(0.104, 0.08, 0.031, 0.08))  # vLLM n-gram, VL run 0005
    assert ngram is not None and "replies 7% shorter, per output token +3% (within noise)" in ngram
    assert shorter_replies(found(0.253, 0.226, 0.211, 0.225)) is None  # Flash-Next smoke-rc7
    assert shorter_replies(found(0.25, 0.17, 0.12, 0.05, "better")) is None
    step = {
        **_step(change=0.104),
        "verdict": "better",
        "phases": {},
        "reason": f"Tensward does not recommend keeping this change: {ngram}.",
        "per_token": {"change": 0.031, "band": 0.08, "verdict": "within_noise"},
    }
    assert _not_better(step).startswith(
        "Requests/s: paired +10%, but replies 7% shorter and per output token +3% (inside its 8%"
    )
