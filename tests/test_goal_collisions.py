import dataclasses

from tensward.classify import classify
from tensward.context import AnalysisContext, active
from tensward.engines import ENGINES
from tensward.engines.llamacpp import predicates as lp
from tensward.measurement import Measurement
from tensward.playbook import Situation, WorkloadFacts
from tensward.settings import Settings

L = ENGINES["llamacpp"]
FACTS = WorkloadFacts(prompts=("hello world " * 300,), output_tokens=512, context_limit=32768,
                      max_inflight=16, load="16 concurrent clients")  # fmt: skip
START = """\
0.05.377.206 I llama_context: n_ctx                 = 32768
0.05.377.210 I llama_context: kv_unified            = false
0.05.377.211 I llama_context: n_ubatch              = 512
0.05.590.661 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 8192
"""
GPU = {"regime": {"placement": "full_gpu"}, "placement": {}, "fit": {}, "environment": {}}


def _llamacpp(tmp_path, m, goal, slots=4, placement=GPU):
    (tmp_path / "server.log").write_text(START)
    settings = Settings(max_concurrent_requests=slots)
    context = AnalysisContext(engine="llamacpp", new_run=tmp_path, goal=goal, placement=placement)
    with active(context):
        diagnosis = classify(m, settings, None, engine=L)
    return Situation(measurement=m, facts=FACTS, settings=settings, diagnosis=diagnosis), context


def test_a_queued_ttft_still_gets_a_larger_ubatch(tmp_path):
    queued = Measurement(200, 0, request_throughput=2.0, output_throughput=600.0,
                         ttft_p50_ms=9000, ttft_p95_ms=15000, tpot_p50_ms=25, tpot_p95_ms=30,
                         queue_share=0.7, peak_waiting=12, peak_running=4, peak_in_flight=16,
                         mean_running=4, mean_waiting=12, mean_in_flight=16,
                         max_prompt_tokens=3000)  # fmt: skip
    situation, context = _llamacpp(tmp_path, queued, "ttft")
    assert situation.finding("prefill").state == "clear"
    with active(context):
        assert "-ub 512 to 2048" in lp.raise_ubatch_applies(situation, L)
    context = AnalysisContext(engine="llamacpp", new_run=tmp_path, goal="throughput", placement=GPU)
    with active(context):
        assert lp.raise_ubatch_applies(situation, L) is None
    all_queue = dataclasses.replace(queued, queue_share=0.99)  # prefill 90 ms of a 7.6 s answer
    situation, context = _llamacpp(tmp_path, all_queue, "ttft")
    with active(context):
        assert lp.raise_ubatch_applies(situation, L) is None


def test_a_queued_ttft_with_short_prompts_keeps_the_expert_offload_reason(tmp_path):
    short = Measurement(200, 0, request_throughput=2.0, output_throughput=600.0,
                        ttft_p50_ms=9000, ttft_p95_ms=15000, tpot_p50_ms=25, tpot_p95_ms=30,
                        queue_share=0.7, peak_waiting=12, peak_running=4, peak_in_flight=16,
                        mean_running=4, mean_waiting=12, mean_in_flight=16,
                        max_prompt_tokens=300)  # fmt: skip
    experts = {**GPU, "regime": {"placement": "expert_offload"}}
    situation, context = _llamacpp(tmp_path, short, "ttft", placement=experts)
    with active(context):
        assert lp.raise_ubatch_applies(situation, L).startswith("the experts run from host memory")


def test_one_client_is_load_limited_for_throughput_but_not_for_a_latency_goal():
    alone = Measurement(60, 0, request_throughput=0.2, output_throughput=20.0,
                        ttft_p50_ms=3000, ttft_p95_ms=3200, tpot_p50_ms=25, tpot_p95_ms=28,
                        queue_share=0.0, peak_waiting=0, peak_running=1, peak_in_flight=1,
                        mean_running=1, mean_waiting=0, mean_in_flight=1, mean_kv_usage=0.3,
                        prompt_work_share=0.3, max_prompt_tokens=3000)  # fmt: skip
    for goal, primary in (("throughput", "load_limited"), ("ttft", "prefill")):
        with active(AnalysisContext(engine="llamacpp", goal=goal)):
            diagnosis = classify(alone, Settings(max_concurrent_requests=1), None, engine=L)
        assert diagnosis.primary == primary
        assert "load_limited" in (diagnosis.primary, *diagnosis.secondary)


def test_a_ttft_named_prefill_keeps_the_small_step_budget():
    from tensward.engines.vllm import predicates as vp

    vllm = ENGINES["vllm"]
    slow_first = Measurement(200, 0, request_throughput=4.0, output_throughput=400.0,
                             ttft_p50_ms=1500, ttft_p95_ms=2500, tpot_p50_ms=20, tpot_p95_ms=25,
                             queue_share=0.05, peak_waiting=0, peak_running=8, peak_in_flight=8,
                             max_prompt_tokens=400, step_budget_tokens=2048, peak_kv_usage=0.3,
                             preemptions=0)  # fmt: skip
    settings = Settings(max_concurrent_requests=8)
    with active(AnalysisContext(engine="vllm", goal="ttft")):
        diagnosis = classify(slow_first, settings, None, engine=vllm)
        situation = Situation(measurement=slow_first, facts=FACTS, settings=settings,
                              diagnosis=diagnosis)  # fmt: skip
        assert situation.fired("prefill")
        assert vp.small_step_budget_applies(situation, vllm)


SPLIT_NP8 = """\
0.05.377.206 I llama_context: n_ctx                 = 98304
0.05.377.210 I llama_context: kv_unified            = false
0.05.377.211 I llama_context: n_ubatch              = 512
0.05.590.661 I srv    load_model: initializing, n_slots = 8, n_ctx_slot = 12288
"""


def test_a_slot_queue_gets_the_trimmed_context_spent_on_more_slots(tmp_path):
    # LC-CTX's parent: -c 98304 -np 8, 24 clients queued for a slot, KV peak 0.13
    queued = Measurement(200, 0, request_throughput=2.0, output_throughput=500.0,
                         ttft_p50_ms=9000, ttft_p95_ms=15000, tpot_p50_ms=20, tpot_p95_ms=25,
                         queue_share=0.9, peak_waiting=16, peak_running=8, peak_in_flight=24,
                         mean_running=8, mean_waiting=16, mean_in_flight=24, peak_kv_usage=0.13,
                         max_prompt_tokens=2700, max_context_len=12288)  # fmt: skip
    (tmp_path / "server.log").write_text(SPLIT_NP8)
    settings = Settings(max_concurrent_requests=8, max_context_len=98304)
    context = AnalysisContext(engine="llamacpp", new_run=tmp_path, goal="throughput", placement=GPU)
    with active(context):
        diagnosis = classify(queued, settings, None, engine=L)
        situation = Situation(measurement=queued, facts=FACTS, settings=settings,
                              diagnosis=diagnosis)  # fmt: skip
        trim = {e.name: e for e in L.playbook()}["llamacpp-trim-max-context-len"]
        assert "16 slots" in trim.applies(situation)
        proposed = trim.apply(situation)
    assert (proposed.max_concurrent_requests, proposed.max_context_len) == (16, 3328 * 16)
    passing = dataclasses.replace(queued, queue_share=0.01, peak_waiting=1, mean_waiting=0.02,
                                  mean_in_flight=8.02, peak_in_flight=9)  # fmt: skip
    with active(context):
        diagnosis = classify(passing, settings, None, engine=L)
        situation = Situation(measurement=passing, facts=FACTS, settings=settings,
                              diagnosis=diagnosis)  # fmt: skip
        assert not situation.fired("queueing") and trim.applies(situation) is None


def test_a_dropped_shared_prefix_leads_for_throughput_on_long_answers(tmp_path):
    from types import SimpleNamespace

    from tensward.suggest import LEAD_SAVING

    long_answers = Measurement(200, 0, request_throughput=1.0, output_throughput=1000.0,
                               ttft_p50_ms=2000, ttft_p95_ms=2500, tpot_p50_ms=25,
                               tpot_p95_ms=30, queue_share=0.05, peak_waiting=0, peak_running=8,
                               peak_in_flight=8, max_prompt_tokens=3000,
                               prefix_cache_hit_rate=0.05)  # fmt: skip
    situation, _ = _llamacpp(tmp_path, long_answers, "throughput", slots=8)
    assert situation.finding("prefill").state == "clear"
    prompts = tuple("you are a careful assistant " * 150 + f"question {i}" for i in range(8))
    situation = Situation(measurement=long_answers, diagnosis=situation.diagnosis,
                          settings=situation.settings,
                          facts=WorkloadFacts(prompts=prompts, output_tokens=1000,
                                              context_limit=32768))  # fmt: skip
    gemma = SimpleNamespace(attention=(SimpleNamespace(kind="sliding"),))
    context = AnalysisContext(engine="llamacpp", new_run=tmp_path, goal="throughput",
                              placement=GPU, anatomy=gemma)  # fmt: skip
    swa = {e.name: e for e in L.playbook()}["swa-full"]
    with active(context):
        assert swa.applies(situation)
        assert swa.leads is not None and swa.leads(situation).share >= LEAD_SAVING


PLACED = {"host_layers": 0, "host_expert_layers": 0, "host_weight_bytes": 0,
          "device_weight_bytes": 2 * 2**30, "host_bytes_per_token": 0,
          "device_bytes_per_token": 2 * 2**30}  # fmt: skip
CPU = {"regime": {"placement": "cpu_only", "compute": "cpu", "host_memory_tight": False,
                  "numa_split": False},
       "placement": PLACED, "fit": {},
       "environment": {"platform": "cpu", "usable_cpus": 4, "smt": False}}  # fmt: skip


def test_more_threads_than_cpus_stays_in_try_first_for_throughput(tmp_path):
    from tensward.suggest import suggestions

    (tmp_path / "server.log").write_text(
        START + "0.05.400.000 I system_info: n_threads = 16 (n_threads_batch = 16) / 16 |\n"
    )
    steady = Measurement(200, 0, request_throughput=0.5, output_throughput=50.0,
                         ttft_p50_ms=900, ttft_p95_ms=1200, tpot_p50_ms=80, tpot_p95_ms=90,
                         queue_share=0.0, peak_waiting=0, peak_running=4, peak_in_flight=4,
                         mean_running=4, mean_waiting=0, mean_in_flight=4,
                         max_prompt_tokens=600)  # fmt: skip
    settings = Settings(max_concurrent_requests=4)
    context = AnalysisContext(engine="llamacpp", new_run=tmp_path, goal="throughput", placement=CPU)
    with active(context):
        diagnosis = classify(steady, settings, None, engine=L)
        assert {f.bottleneck: f.state for f in diagnosis.findings}["host_overhead"] == "cant_tell"
        situation = Situation(measurement=steady, facts=FACTS, settings=settings,
                              diagnosis=diagnosis)  # fmt: skip
        offered = suggestions(L, situation, current=settings, project_path=None).suggestions
    assert {s.entry.name: s.tier for s in offered}["cpu-threads"] == "try_first"


def test_thread_offers_use_every_vcpu_when_offloaded(tmp_path):
    from tensward.engines.llamacpp import predicates as P

    smt = {**CPU, "environment": {"platform": "nvidia", "usable_cpus": 26, "smt": True,
                                  "physical_cores": 13},
           "regime": {**CPU["regime"], "placement": "expert_offload",
                      "compute": "mixed"}}  # fmt: skip
    situation = Situation(measurement=Measurement(10, 0), facts=FACTS, settings=Settings(),
                          diagnosis=None)  # fmt: skip
    for threads, offered in ((13, 26), (6, 26), (26, None)):
        run = tmp_path / str(threads)
        run.mkdir()
        (run / "server.log").write_text(
            START + f"0.05.400.000 I system_info: n_threads = {threads} (n_threads_batch = "
            f"{threads}) / 26 |\n"
        )
        with active(AnalysisContext(engine="llamacpp", new_run=run, placement=smt)):
            assert bool(P.threads_smt_applies(situation, L)) == (offered is not None)
            if offered:
                settings = P.with_threads(situation, L)
                assert settings.extra_args["--threads"] in (offered, str(offered))


def test_cuda_graphs_off_stays_in_try_first_for_throughput_without_a_trace():
    from tensward.suggest import suggestions

    vllm = ENGINES["vllm"]
    eager = Measurement(200, 0, request_throughput=2.0, output_throughput=500.0,
                        ttft_p50_ms=60, ttft_p95_ms=90, tpot_p50_ms=8, tpot_p95_ms=10,
                        queue_share=0.0, peak_waiting=0, peak_running=4, peak_in_flight=4,
                        mean_running=4, mean_waiting=0, mean_in_flight=4, max_prompt_tokens=300,
                        peak_kv_usage=0.1, preemptions=0, step_budget_tokens=2048,
                        max_context_len=32768)  # fmt: skip
    settings = Settings(max_concurrent_requests=4, cuda_graphs=False, max_context_len=32768)
    with active(AnalysisContext(engine="vllm", goal="throughput")):
        diagnosis = classify(eager, settings, None, engine=vllm)
        situation = Situation(measurement=eager, facts=FACTS, settings=settings,
                              diagnosis=diagnosis)  # fmt: skip
        offered = suggestions(vllm, situation, current=settings, project_path=None).suggestions
    assert {s.entry.name: s.tier for s in offered}["cuda-graphs"] == "try_first"
    graphs = dataclasses.replace(settings, cuda_graphs=True)  # decode bandwidth is untold here
    with active(AnalysisContext(engine="vllm", goal="throughput")):
        diagnosis = classify(eager, graphs, None, engine=vllm)
        situation = Situation(measurement=eager, facts=FACTS, settings=graphs,
                              diagnosis=diagnosis)  # fmt: skip
        offered = suggestions(vllm, situation, current=graphs, project_path=None).suggestions
    assert {s.entry.name: s.tier for s in offered}["ngram-speculation"] == "could_help"


T2_START = """\
0.02.399.228 I load_tensors:        CUDA0 model buffer size = 14674.45 MiB
0.05.917.494 I llama_context: n_ctx                 = 93184
0.05.917.495 I llama_context: n_ubatch              = 512
0.05.917.496 I llama_context: kv_unified            = true
0.05.948.279 I llama_kv_cache:      CUDA0 KV buffer size =  5824.00 MiB
0.05.948.296 I llama_kv_cache: size = 5824.00 MiB ( 93184 cells,  16 layers,  4/1 seqs), K (f16): 2912.00 MiB, V (f16): 2912.00 MiB
0.06.052.453 I sched_reserve:      CUDA0 compute buffer size =   234.44 MiB
0.06.347.835 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 93184, kv_unified = 'true'
"""  # noqa: E501
T2_FIT = {"compute_buffer_bytes": 245828157, "device_free": 23455596544,
          "device_need": 22813752688, "fits": True, "host_free": 64375558144, "host_need": 0,
          "kv_bytes": 6106906624, "need_tokens": 0, "peak_running": 1, "pool_tokens": 93184,
          "reason": "both domains fit", "slots": 4, "tight": False, "ubatch": 512}  # fmt: skip
T2_PLACED = {"device_bytes_per_token": 15387276083, "device_weight_bytes": 15387276083,
             "host_bytes_per_token": 2879, "host_expert_bytes": 0, "host_expert_layers": 0,
             "host_layers": 0, "host_weight_bytes": 0, "kv_on_host": False}  # fmt: skip


# T2's own log: --fit on chose the 93184-token context
T2_FIT_LOG = """\
0.00.892.761 I common_params_fit_impl: projected to use 32058 MiB of device memory vs. 22369 MiB of free device memory
0.00.892.779 I common_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce device memory by 10713 MiB
0.01.434.413 I common_params_fit_impl: context size reduced from 262144 to 93184 -> need 10727 MiB less memory in total
0.01.434.427 I common_params_fit_impl: entire model can be fit by reducing context
"""  # noqa: E501


def _t2(tmp_path, settings, log=T2_START, output_tokens=1024):
    """T2: --ctx-size 93184 --parallel 4 --kv-unified, prompts up to 7,218 tokens, 612 MiB left."""
    (tmp_path / "server.log").write_text(log)
    regime = {"placement": "full_gpu", "compute": "gpu", "host_memory_tight": False,
              "numa_split": False}  # fmt: skip
    block = {"regime": regime, "fit": T2_FIT, "placement": T2_PLACED,
             "environment": {"platform": "nvidia"}}  # fmt: skip
    t2 = Measurement(39, 1, request_throughput=0.0256, output_throughput=11.8, ttft_p50_ms=7316,
                     ttft_p95_ms=8974, tpot_p50_ms=69.5, tpot_p95_ms=69.8, queue_share=0.05,
                     peak_waiting=0, peak_running=1, peak_in_flight=1, mean_running=1,
                     mean_waiting=0, mean_in_flight=1, max_prompt_tokens=7218,
                     max_context_len=93184)  # fmt: skip
    facts = WorkloadFacts(prompts=("x " * 5000,), output_tokens=output_tokens, context_limit=262144)
    context = AnalysisContext(engine="llamacpp", new_run=tmp_path, goal="ttft", placement=block)
    with active(context):
        diagnosis = classify(t2, settings, None, engine=L)
    return Situation(measurement=t2, facts=facts, settings=settings, diagnosis=diagnosis), context


def test_a_reserved_context_buys_the_room_a_larger_ubatch_was_refused(tmp_path):
    settings = Settings(max_concurrent_requests=4, max_context_len=93184,
                        extra_args={"--kv-unified": True})  # fmt: skip
    situation, context = _t2(tmp_path, settings)
    with active(context):
        assert situation.fired("prefill") and lp.raise_ubatch_applies(situation, L) is None
        trade = {e.name: e for e in L.playbook()}["context-for-room"]
        why = trade.applies(situation)
        proposed = trade.apply(situation)
    assert "needs about 703 MiB more" in why and "only 612 MiB is left" in why
    assert "reserves a KV pool any one request can fill" in why
    assert "--ctx-size 91648 frees about 96 MiB" in why and "none past 91,648 tokens" in why
    assert "documented, not measured" in why
    assert proposed.max_context_len == 91648 and proposed.extra_args["--ubatch-size"] == 2048


def test_a_split_pool_cut_names_the_tokens_each_slot_keeps(tmp_path):
    split = T2_START.replace("kv_unified            = true", "kv_unified            = false")
    split = split.replace("n_ctx_slot = 93184, kv_unified = 'true'", "n_ctx_slot = 23296")
    settings = Settings(max_concurrent_requests=4, max_context_len=93184)
    situation, context = _t2(tmp_path, settings, split)
    with active(context):
        why = {e.name: e for e in L.playbook()}["context-for-room"].applies(situation)
    assert "reserves 4 slots of 23,296 tokens" in why and "none past 22,912 tokens" in why


def test_a_context_fit_chose_gives_a_larger_ubatch_its_room(tmp_path):
    # T2 as launched: --kv-unified --fit on and no --ctx-size; fit cut 262144 to 93184
    settings = Settings(max_concurrent_requests=4, extra_args={"--fit": "on", "--kv-unified": True})
    situation, context = _t2(tmp_path, settings, T2_FIT_LOG + T2_START, output_tokens=1280)
    entries = {e.name: e for e in L.playbook()}
    with active(context):
        why = lp.raise_ubatch_applies(situation, L)
        proposed = lp.with_larger_ubatch(situation, L)
        assert entries["context-for-room"].applies(situation) is None
        assert lp.raise_ubatch_blocked(situation, L) is None
    assert why.startswith("TTFT p50 7316 ms is 105x TPOT p50 69.5 ms; -ub 512 to 2048")
    assert "fit picks a context about 1,536 tokens smaller than today's 93,184" in why
    assert proposed.extra_args["--ubatch-size"] == 2048 and proposed.max_context_len is None


def test_a_limit_whose_change_lacks_room_says_how_much_is_missing(tmp_path):
    from tensward.report.build import next_steps_section
    from tensward.report.markdown import sections_markdown
    from tensward.suggest import suggestions

    settings = Settings(max_concurrent_requests=4, extra_args={"--kv-unified": True})
    situation, context = _t2(tmp_path, settings)  # the context is the engine's: nothing to trade
    with active(context):
        offered = suggestions(L, situation, current=settings, project_path=None)
        text = sections_markdown([next_steps_section(situation.diagnosis, offered, retained=True)])
    group = text.split("For prefill compute:")[1].split("\n\n")[0]
    assert "-ub 2048 would help but needs about 703 MiB more GPU memory" in group
    assert "no change in this engine's playbook applies here" not in group
    assert "q8_0" not in group
    assert "Nothing in Try first can run for TTFT: the change for prefill compute" in text


def test_a_blocked_change_is_shown_beside_the_offers_of_its_group(tmp_path):
    from dataclasses import replace

    from tensward.report.build import next_steps_section
    from tensward.report.markdown import sections_markdown
    from tensward.suggest import Suggestion, suggestions

    settings = Settings(max_concurrent_requests=4, extra_args={"--kv-unified": True})
    situation, context = _t2(tmp_path, settings)
    with active(context):
        offered = suggestions(L, situation, current=settings, project_path=None)
        entry = next(e for e in L.playbook() if "prefill" in e.addresses)
        other = Suggestion(entry=entry, reason="r", tier="try_first", addresses=entry.addresses,
                           basis=entry.basis, settings=settings,
                           command="tensward try")  # fmt: skip
        text = sections_markdown(
            [next_steps_section(situation.diagnosis, replace(offered, suggestions=(other,)),
                                retained=True)]
        )  # fmt: skip
    assert "-ub 2048 would help but needs about 703 MiB more GPU memory" in text


def test_the_room_a_refused_change_needs_comes_from_a_lever_that_frees_it():
    def view(kind):
        return lp.View(started=None, block={"regime": {"placement": kind}}, probes={})

    assert "--n-cpu-moe" in lp._free_room(view("expert_offload"))
    assert "--ctx-size if your requests need less" in lp._free_room(view("expert_offload"))
    assert "--n-gpu-layers" in lp._free_room(view("dense_offload"))
    assert lp._free_room(view("gpu_only")) == lp.FREE_ROOM
