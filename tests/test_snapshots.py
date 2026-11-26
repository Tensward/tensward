from pathlib import Path

from tensward.engines.llamacpp.snapshots import Snapshots, read

HERE = Path(__file__).resolve().parent


def _log(restores: dict[int, list[int]], served: int = 3) -> str:
    lines = [
        "0.01.000.000 I srv    load_model: context checkpoints enabled, max = 32, "
        "min spacing = 8192"
    ]
    for slot in range(8):
        for _ in range(served):
            lines.append(
                f"0.02.000.000 I slot   operator(): id  {slot} | task 1 | new prompt, "
                "task.n_tokens = 640"
            )
        for n in restores.get(slot, []):
            lines.append(
                f"0.03.000.000 I slot   operator(): id  {slot} | task 2 | restored context "
                f"checkpoint (n_tokens = {n}, n_past = {n})"
            )
    return "\n".join(lines)


def test_the_shared_start_and_the_slots_that_never_resume_there_come_from_the_log() -> None:
    """A trimmed excerpt of a b12441 -lv 4 log: Qwen3.8-27B on an L4, -np 8 -b 512."""
    found = read((HERE / "snapshots" / "llamacpp_b12441_l4_b512_8slots.log").read_text())
    assert found == Snapshots(566, 5, 8, 8192)


def test_a_resume_past_the_start_does_not_count_and_two_slots_make_a_start() -> None:
    restores = {slot: [566] for slot in (0, 1, 2, 3, 5, 7)} | {4: [510, 536], 6: [255, 659]}
    assert read(_log(restores)) == Snapshots(566, 2, 8, 8192)
    assert read(_log({0: [566], 1: [300]})).shared_start_tokens is None
    assert read(_log({})) == Snapshots(None, None, 8, 8192)
    assert read(_log({s: [566] for s in range(8)}, served=1)).lost == 0


def test_only_a_similarity_pick_followed_by_a_full_read_counts() -> None:
    from tensward.engines.llamacpp.snapshots import similarity_rereads

    pick = "0.01.000.000 I slot       select: id  0 | task -1 | selected slot by {} thold"
    forced = "0.01.000.001 I slot   operator(): id  0 | task 3 | forcing full prompt re-processing"
    log = "\n".join([pick.format("LRU"), forced, pick.format("LCP similarity"), forced, forced])
    assert similarity_rereads(log) == 2
    assert similarity_rereads(forced) is None
