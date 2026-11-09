from types import SimpleNamespace

from tensward.toolcheck import PROBE_MAX_TOKENS, probe_tokens


def test_the_probe_has_the_room_the_workload_gives_a_thinking_model():
    thinking = SimpleNamespace(config=SimpleNamespace(workload=SimpleNamespace(output_tokens=1280)))
    short = SimpleNamespace(config=SimpleNamespace(workload=SimpleNamespace(output_tokens=128)))
    assert probe_tokens(thinking) == 1280  # type: ignore[arg-type]
    assert probe_tokens(short) == PROBE_MAX_TOKENS  # type: ignore[arg-type]
