# Changelog

## 0.3.16

Safer and faster steps, stacks of up to four changes, and a place for what is known about each
change.

Changes that can affect what Tensward recommends:

- An answer check that timed out or failed to return a result leaves its step available once
  more. The suggested command uses `--reuse-run` to copy the measured run and its answers
  when they are still eligible, or measures again otherwise. A second rejection drops it.
- Small lossless changes are bundled into one full-window step. A bundle within noise can be
  kept; a worse result brings its members back as single steps, and a start failure removes
  the member named by the engine.
- On a text-only workload for a checkpoint with media encoders, the first stack or bundle
  starts with `--language-model-only` and an explicit KV size when the pool holds at least
  what its parent used. Otherwise it keeps the existing setup.
- A step whose KV pool would be under half of what its parent used is stopped within seconds,
  recorded in its `window.json`, and offered again with an explicit KV size. A server with no
  memory left for its KV cache says so. A step that queues in its first minute where its parent
  did not is stopped and recorded the same way, and is never "the latest run".
- Text-only serving always carries an explicit KV size; without one it limits image, video and
  audio to zero. The size is worked out from the engine's own figure, never from a sampled peak
  alone.
- New opt-in steps: an explicit KV size; per-channel fp8 weights, used when the KV cache keeps
  headroom; a smaller step budget; KV-sharing fast prefill for models with KV-shared layers, with
  the answers compared. fp8 weights are also offered on a saturated GPU that computes in fp8
  when prompts take a large share of its work.
- n-gram speculation runs on the GPU (`ngram_gpu`). The tool-call parser is picked for Qwen3,
  Granite and gpt-oss. Checkpoints in mxfp4 are accepted, multi-head latent attention and hybrid
  models are read, and the vLLM 0.30 memory gates are applied in the fit.
- Checkpoints using GPTQ act-order are refused on vLLM 0.30, before any server starts.
- A project whose configuration turns tool calling on while its `--current` command does not
  enable it is refused at `init` and at `analyse`, with the flags to add.
- Judged steps (those that can change the answers) measure at least 150 requests, even above the
  configured count, as far as the prompt file reaches: the floor is the smaller of 150 and the
  larger of the configured count and the file's prompts. The requests above the configured count
  repeat prompts the run already sent, so such a step takes longer than the configured count
  suggests.
- Under the ttft goal the first run plans enough requests to read TTFT after the ramp (more
  clients need more requests), and a step from it sends at least as many. A TTFT p95 read after
  the ramp says so in the verdict; when too few requests remain after the ramp, it says it was
  read over the full matched window.
- A KV-bound throughput step is measured on a shorter steady window, after a bias check. A count
  per client plans the seconds it takes; an explicit `--window SECONDS` is never raised.
- Answers of runs that sent different prompts are compared on the prompts both sent.
- Offers stack two to four changes, "measured together", ordered by predicted gain against
  answer risk, with the lossless remainder listed apart. `report.json` shows `answer_risk`. A
  stack that Tensward would not keep as within noise gives way to its lossless remainder, and a
  remainder the judge turned down ends the stack.
- Under the tpot goal, on a run whose KV space held requests back, fp8 weights (with fp8 KV where
  the pair is offered) and a concurrency cap that keeps the batch near where it ran are offered
  as one "held batch" step. With tool calls the fp8 weights and fp8 KV pair is one step only where
  KV space held requests back, and on a checkpoint under 2B parameters it waits under Could
  help. A held batch that the start-up guard or the load watcher stopped is not offered again.
- A Try-first offer predicted to worsen the goal moves to Could help with the figure. Try first
  lists changes that can affect answers first, then long starts, then short starts and cache
  hits. Start estimates use this project's measured starts of the same class. When Try first
  has nothing left to run, the lossless Could-help offers move up ("measured now: nothing else
  in Try first can run").
- Under ttft, where the tool-call parser holds the first content back for 4 or more decode
  steps, offers rank on the engine's own time to first token, and the report says so.
  `--ttft-from first-token` ranks on the first streamed chunk instead. Measured TTFT is
  unchanged. Where the wait is queue time, the client's figure stays and the report says why.
- A raise from the engine's default (the memory share, the step budget, the concurrency cap) is
  read as up or down against that default.
- A smaller step budget is no longer offered when first tokens wait mostly in the queue: the
  budget only acts once a request is scheduled.
- An explicit KV size is no longer offered when requests wait only for the concurrency cap; it
  needs the waiting to come from KV space.
- A full-window step (such as a bundle) is timed from this run's warm-up and full measurement,
  so its `test_seconds` no longer plans the shorter automatic window.

What else is new:

- Every run writes `facts.json`, the situation as data. `report.json` has a `kb` key
  (`{"pack": null}` without an installed knowledge source). A knowledge source installed beside
  Tensward adds a "Why" line under the offers its records explain, and a "Worth measuring"
  section for records that say what could help but have not been measured here. `tensward report
  --kb` prints the records behind each offer. Your own records live under `TENSWARD_KB_HOME`
  (default `~/.tensward/kb`); a measured step appends one training row there. With nothing
  installed and no local records the offers are the playbook's alone.
- `--engine-env NAME=VALUE` sets an allowed engine environment variable for one run with
  `--runtime local`; steps and kept settings carry it, and `tensward reject` accepts a run that
  changed only the environment.
- `--pin-client` keeps the engine off one CPU core so the load generator has a core of its own.
- `--borrow-reference` reuses a reference run of the same setup up to 14 days old that ran under
  this machine's power cap, on this machine's own GPUs (not an imported run); a step from it is judged with the cross-launch floor widened.
- `window.json` records which compile caches were warm, and the start-up phases and start class.
- With `--runtime local`, `TENSWARD_ENGINE_LAUNCHER` wraps the server command. A launcher ending
  in `--` receives the engine's added environment as `--env NAME=VALUE` pairs before it.
- Suggestion blocks in `report.json` include `test_seconds` (launch to the end of measurement)
  and `start_seconds` (launch to ready for that step's start class), or null when unknown.
- `requests.jsonl` records the time of the first streamed chunk with a choice. An int8
  per-token-head KV cache set by hand counts as an 8-bit cache: its answers are compared.

Left for 0.4.x:

- Matching a run's facts against the records takes about 32 ms once warm, over the design's
  10 ms; the first call takes 45 ms. Making it faster is deferred, and so is the warm re-measure
  of a step (measuring it again once the engine is warm).
- Borrowing a knowledge-base measurement as the probe of an engine setting, and re-reading the
  maximum model length that the fit derives, are deferred.

## 0.3.13 (unreleased)

- Checkpoints with scalar tensors (for example Gemma 4 E4B's encoders) are read instead of
  refused.

## 0.3.12 (unreleased)

Faster steps, and verdicts that say how sure they are.

Changes that can affect what Tensward recommends:

- A step (an `analyse` with a change, after an earlier run of the project) measures a planned
  window instead of the configured request count: the shortest window whose noise band is narrow
  enough for the goal, never more than configured. `--window auto|full|SECONDS` sets it
  (`full` measures the configured count). The plan and why are on standard error, in
  `run.json` (`analysis.window`) and in the new `window.json`. A run that planned a window is
  not reused as a reference.
- A run short of KV memory (its KV-capacity finding fired, a request was preempted, or the KV
  cache peaked at 95% or more) plans no short window: a step from it, or `--window SECONDS` on
  it, measures the configured count, so the step and its parent compare full against full. On
  an L4 short of KV memory a 60 s window of the unchanged setting read +12% requests per second
  against the full run, beyond its ±7% band.
- Every verdict against the run a step was made from, "What changed" rows and the keep section
  included, shows its noise band ("TPOT p95 -8% ±3%"), measured from the two runs' own requests
  on the prompts both sent. A move inside the band reads "within noise", and a loss or a lead
  counts only beyond it. When the answers of a short step are compared, only the prompts both
  runs sent are counted ("of the N prompts both runs sent").
- Tensward does not recommend keeping a change within noise. `--adopt-within-noise`, or
  `TENSWARD_ADOPT_WITHIN_NOISE=1`, is an opt-in for experiments: it adopts a step whose point
  estimate is better on the goal when nothing you limited worsened beyond its band. The report
  says so, and `run.json` and `report.json` record it.
- A plain `analyse` reuses the newest trusted run of the same setup instead of starting the
  engine again (same engine version, GPU and driver, the full window, at most 2% failed requests,
  measured within 24 hours); the report names the run. `--remeasure` measures again.
  `tensward import RUN_DIR --project P` copies a run of another project with the same
  registration so that it can be reused.
- A memory-traffic model predicts the effect of fp8 weights, the KV-cache precision and a
  concurrency cap from the run's own decode step, and the prediction is shown with the offer.
  Where the prompt share of the work was not measured, the prediction says "prompt share at
  most X%". An offer predicted to cross your requests-per-second floor or a declared TPOT limit
  moves to Could help.
- Try first starts with the offer predicted to gain the most on the goal (at least 10%), inside
  your floor and limits.
- Two changes that do not interact are measured together in one step ("measured together"), from
  a table of known pairs, when your declared limits leave room for both. If the pair gains less
  than the first change was predicted to alone, the report says so and offers that change alone
  from the same starting settings. Rejecting the pair brings both back as single offers.
- New offer `fp8-kv-decode`: an 8-bit KV cache where decode reads mostly KV cache (KV bytes read
  per step at least twice the weight bytes, decode at 60% or more of its memory-bandwidth
  ceiling). It is never offered for Qwen2 or Qwen2.5 models, on a GPU without fp8 KV, on a hybrid
  cache or when the cache is already 8-bit, and it shows the answers before and after. Where KV
  memory is short under a goal other than TPOT, the existing `fp8-kv-cache` offer stays the only
  one.
- N-gram speculation is offered in Could help as "measure it" with the replay's coverage, not a
  break-even claim: its step overhead differs too much between models. Under the TPOT and
  throughput goals a pooled replay below 6% leaves it out.
- On runs whose requests make tool calls, where the time to first token is mostly spent waiting
  behind decode steps, the prediction for a precision change includes its effect on TTFT. Under
  `--goal ttft` Try first is ordered by the predicted TTFT change.
- When KV memory limits how many requests run at once, the predictions for fp8 weights and an
  8-bit KV cache count the larger batch the freed memory holds, up to the concurrency cap and
  the demand. A run already serving fp8 weights is predicted from its real weight bytes.
- On such a KV-bound run without tool calls, fp8 weights and an 8-bit KV cache are offered as one
  step, with the prediction for both, unless that prediction crosses a declared limit.
- When Try first would hold no runnable change, one safe change from Could help moves up, and
  its reason says there is no strong signal for it.
- On runs with tool calls, Try first puts the 8-bit KV cache before fp8 weights: 8-bit weights
  also quantize the activations, which cost tool-call validity on measured runs.
- A planned window needs at least 4 requests per client instead of 8, so steps with few clients
  are shorter.

Changes that do not affect recommendations:

- Prompt token counts are kept once per project in `cache/token_ids.json`, so later runs do not
  count them again.
- The read of which quantization methods the engine supports is cached per engine and GPUs; a
  failed read is never cached.
- The docker runtime keeps the engine's compile caches (vLLM, Triton, FlashInfer) per image in
  `~/.cache/tensward/engine-cache/<image id>/`. The container writes them as root; clear them
  with `sudo rm -rf ~/.cache/tensward/engine-cache`.
- vLLM start times (initialisation, compile, CUDA graph capture, warmup) are read from the server
  log and recorded in `window.json`.
- A prompt-capacity probe that fails because of the prompts' shape (too long, too short, or a
  cached prefix) is remembered per model, engine and prompts, so later launches skip it.
- `analyse` loads and parses the project once instead of twice, which shortens the start on
  large workloads.
- `run.json` and `report.json` only gain fields; `schema_version` stays "1".

## 0.3.11 (unreleased)

Changes that can affect answers:

- `analyse` reads which quantization methods and KV-cache types the installed engine supports,
  without loading a model, and intersects them with the GPU's compute capability. What it offers
  comes from that read, not from a fixed list: `fp8-weights` (`quantization=fp8`), `bnb-4bit-weights`
  (only with the `vllm-bnb-plugin` installed), `quantized-checkpoint` (the 8-bit formats the GPU
  runs, then the 4-bit ones; Tensward downloads nothing, and the cost line says how to register
  the checkpoint and compare), and `fp8-kv-cache` only where the GPU runs it. A failed read never
  stops a run; the report says it could not read the support and offers no weight quantization.
  The result is recorded in `run.json` as `answer_options`. vLLM 0.30.0 starts the CUTLASS fp8
  kernel below compute capability 10.0, which fails in the profile run on a T4 and on an A10G
  (both measured). So fp8 weights, made at start or from an fp8 checkpoint, are offered as W8A8
  from 8.9 (from vLLM 0.30.0's source, not run) and as weight-only W8A16 on 8.0-8.8, where the
  command adds `linear-backend=marlin` (Marlin serves them on an A10G, measured); below 8.0 they
  are not offered. The fp8 KV cache is offered only from 8.9 (from vLLM 0.30.0's source, not
  run). On an A10G it did not start for Phi-3.5-mini (head dim 96), and for Qwen2.5-7B (head dim
  128) it started and served, but none of its 600 answers matched bf16's: the mean word overlap
  was 0.34 and 84% shared under half their words, many repeating themselves (measured).
- Every such change says, in words, that answers may change, and 4-bit changes say they change
  more often. Under a goal, changes that cannot affect answers come first. A change that can
  affect them is always shown in full.
- A run whose settings differ from the run it was made from in the weight quantization, KV-cache
  precision, speculation or compact JSON adds "Answers before / after" to its report: how many
  answers differ, with the text answers' mean word overlap, tool calls and text counted apart
  when tools were called, the answers that repeat themselves or are empty before and after, valid
  JSON and failed requests, and the least similar prompts side by side. Every pair, and a
  `summary` with those counts, is written to `quality.json` in the run directory; each pair says
  whether it could be `judged` (the earlier run repeated the prompt). Tensward does not say which
  answers are better.
- Broken answers are flagged, not judged: when the share of answers that repeat themselves (more
  than half of the text repeating itself, as in "be be be") or are empty rises by more than 5
  points, the report says so under its header, the keep section and `tensward set` warn, and the
  comparison's verdict reads "Answers changed". The count is a floor (an answer cut short before
  it repeats is not counted), so the warning says "at least" and adds the answers' mean word
  overlap with the earlier run's. On an A10G, an fp8 KV cache on Qwen2.5-7B left
  53 of 600 answers repeating themselves, against none with bf16 (measured).
- A run that already carries a change Tensward rules out for the model or the GPU (an fp8 KV
  cache on Qwen2 or Qwen2.5, or below compute capability 8.9) is not recommended for keeping (the
  keep section and a refused `tensward set` print the warning first, then the command for one
  who has read the answers and still wants them) and gets no change on top of it; going back to the run it was made from (or to the registered setup)
  comes first. Below 8.9 the fp8 KV cache is left out also when the support read failed.
- `--baseline-answers` accepts a run's own `responses.jsonl`, so a different checkpoint can be
  compared with a run.

Keeping and dropping a change:

- New `tensward set RUN --project P` makes a run's settings the kept setup. `RUN` is a run id,
  `last`, or `registered` to go back to the setup registered with `init`. `analyse` and
  `serve start --from current` then start from the kept setup, and every analyse of it is
  compared with the registered setup's run. A run whose change can affect answers is kept only
  with `--accept-answer-changes` (`--accept-unreviewed` still works, here and in `serve start`).
  `set` prints the settings before and after and the command that undoes it, and says when the
  run carries a change you rejected. While a setup is kept, a revert to the registered setup is
  `tensward set registered --project P`. State is in `<project>/kept.json`.
- Every report ends with "To keep this change": the exact `tensward set` command, what it does and
  how to undo it. Under a latency goal it repeats the measured gains and costs since the starting
  settings in one line. When an earlier run of other settings did better on the goal, the report
  says so, names that run as the best so far and prints its `tensward set` command first; one
  within noise of this run is named after the command.
- New `tensward reject RUN --project P [--reason TEXT]`, `--clear RUN` and `--list`: a rejected
  change is never offered again for the project, and is listed once under So far. The run a
  change was made from is never one that carries a rejected change. State is in
  `<project>/rejected.json`. Rejecting the kept setup, or a change that is part of it, is refused
  and points to `tensward set registered --project P`.
- New `tensward report RUN --project P` writes a run's report again from its run directory, with
  the project's current history, kept setup and rejections: no engine starts and no new run is
  made, so the next suggestion is available after a `reject` or a `set`. It refuses (exit 1) a
  run written before 0.3.11 or whose analysis did not finish, and a run profiled with `--trace` or
  `--counters`; on a directory that is not a registered project it exits 2 with
  `project_not_registered`.

Latency goals:

- New `no-async-scheduling` entry under `--goal ttft`, offered when the engine took two decode
  steps or more to prefill a request. TTFT improves; requests per second and TPOT worsen, and the
  entry says so in words. Its reason quotes one measured cell (a T4 with Qwen3-4B), never a
  prediction for the GPU at hand.
- `small-step-budget` offers 256 tokens per step only when the longest prompt with the running
  decodes fits 256 tokens (it checked 512 before). On a power-capped T4 serving prompts up to
  182 tokens at 128 clients the ladder is 512 tokens per step, async off, then stop, and the stop
  gives TTFT p95 in decode steps.
- `--max-req-s-loss PCT` sets a floor for `--goal ttft|tpot`, judged on measured runs. Once a run
  is more than PCT% below the trade base in requests per second, a change that would lower them
  further moves to Could help with the measured loss as its reason, and going back to the latest
  run within the floor comes first. That run never carries a rejected change, and never has
  settings that can change answers against the registered setup unless they are the kept setup.
  Before that, the cost line names the floor in requests per second and what it is measured
  against. The best run so far is never one below the floor. `tensward set` keeps the floor of
  the run it keeps, so the kept setup's runs and `Try:` commands use it. `report.json` keeps it as
  `goal.floor`.
- The trade line is measured from the latest run of the project's starting settings, not from the
  first run, which starts cold. When the start-up wave got more than 20% slower in TTFT p95, and
  both waves had two requests or more, a second line says so.

Report wording:

- A move whose printed percent is 2% or less in throughput, or 5% or less in a latency, reads
  "within noise" instead of "worse", in "What changed" and in So far, so a printed "+2%" reads the
  same for every metric; a move that rounds to none reads "unchanged".
- Under a latency goal, the bottleneck line says when the run's overall bottleneck does not hold
  back the goal, so it no longer contradicts the goal's line above it.
- `lower-prefill-batch` is no longer offered below its own gate.
- `Try:` commands repeat `--runtime` (and `--image`, `--local-command` or `--gpus` when given), so
  a docker run's next step runs in docker too, on the same GPUs.
- A run of the kept setup is named "kept setup (run R)" in the header and setup section.
- The `fp8-weights` reason is one plain sentence; plan lines no longer say "(it is crossed now)"
  or name a change the report does not offer.
- `tensward set registered` prints what it dropped; a refused `set` names the run's report;
  `--local-command` help says it is the whole command, `serve` included; a failed `serve start`
  names the server's log; the hashing note gives GiB, as `init` does.
- A run whose parent had no plan no longer says "no plan to check from run R".
- The TTFT diagnosis no longer says "none of its signals crossed" before a step-budget change: it
  says no single limit was measured as binding and gives TTFT p95 in decode steps.
- "What changed" and the comparison say again when the two runs started with a warm and a cold
  compile cache (the analyse report had lost the candidate's cache state).
- Smaller fixes: "bottleneck: unchanged (...)", "kept in this run" and "changed to ... in this
  run" for tried changes, no "left out" line for undoing this run's own change, `--reason
  "<why>"` in the reject line, and an `init` warning that says `vllm` is not on PATH (activate
  its environment) instead of a garbled sentence.
- No pre-run KV-cache estimate is printed for weights quantized at start: it counted the
  checkpoint's unquantized weights and read 30-61% low.
- The terminal header says "current setup (declared in config) + this run's overrides: ...";
  with a kept setup, "What changed" and So far name "the registered setup".
- "What changed" reads "answers changed: ..." instead of "answers: Answers changed: ...";
  requests per second below 10 show two decimals there; a left-out change's reason line says
  "is left out"; a bottleneck held back by a revert says so instead of "no change ... applies".

Sizing and speculation:

- The concurrency cap sized from the KV pool uses the pool measured at the same per-step token
  budget when the project has one, else the smallest cold pool as before.
- `ngram-speculation` replays prompt lookup over this run's own answers, without a GPU, and is
  offered only when the share of output tokens the drafts would supply beats the step overhead;
  under `--goal tpot` the answers it helps least decide. The reason states the share and the
  break-even. Where the run cannot be replayed, the reason says why. The step overhead is 4.5 ms,
  measured on an A10G at 32 running requests (Phi-3.5-mini: step 41.0 to 45.5 ms, TPOT p95 +8%;
  the replay supplied 3.7% of the slowest answers' output, where 9.9% would be needed). Under
  `--goal tpot`, when the replay predicts a worse TPOT, the entry is listed under "left out
  because they worsen TPOT" with the replay.
- A prediction about decode memory bandwidth, checked on a run that speculates, now reads "can't
  tell": Tensward does not model the decode ceiling while speculating.

Capacity at a limit:

- New `tensward analyse --capacity` with `--slo-ttft-p95` and/or `--slo-tpot-p95` measures steps of
  closed-loop clients on one launch and reports the highest requests per second whose TTFT p95 and
  TPOT p95 are within the declared limits. The figure is measured, not estimated, and the output
  lists every step. When the first step outside missed a latency limit by 10% or less, a line
  says the capacity lies between the two steps' measured figures. Without a declared limit it
  refuses, in one sentence.
- `--slo-tpot-p95` is the new name of `--slo-tpot-ms`, which still works.

Output:

- Additive only: `answers_after` and `keep` sections, `goal.floor`, `goal.trade.wave` and
  `goal.check.cant_tell` in `report.json`; `answer_options` and `analysis` in `run.json`;
  `quality.json`; `kept.json`, `rejected.json` and `capacity/` in the project directory; a `kept`
  key in the `init` and `inspect` JSON while a setup is kept; an "answers that repeat themselves
  (empty ones not counted)" row in the comparison's quality table, when either side has one. API
  version 1 is unchanged; `Entry` gains `changes_answers` and `answer_bits`.
- `prompt_blocks.jsonl` is written for every run whose engine returned the prompts' token ids, not
  only when `prefix-caching` was offered, so `tensward report` can replay the shared blocks.
- The tried line for a change a later run still carries now reads "still in this run" instead of
  "kept", since `tensward set` now means keeping.
- `--local-command` and `--image`, repeated in every suggested command, and the server command in
  `run.json` have the values of secret-named variables and flags blanked (`HF_TOKEN=***`).
  `CREDENTIAL(S)`, `PASS` and `BEARER` now count as secret names, alongside `KEY`, `TOKEN`,
  `SECRET`, `PASSWORD` and `AUTH`.
- Answer excerpts in the report and the answers file have control characters and terminal escape
  sequences removed, and the report shows them as code; `quality.json` keeps them as returned.
- A failed `serve start` saves the server's log with the API key replaced by `***`, as `analyse`
  does.
- The `tensward set registered` and `tensward analyse` commands shown in messages quote the
  project path.

Measured with vLLM 0.30.0, one run per step; each figure holds for its own setup only:

- Tesla T4 at its 70 W cap, Qwen3-4B, short Q&A prompts, 128 closed-loop clients, `--goal ttft`
  (one cell). Turning async scheduling off at 512 tokens per step, in two sessions: TTFT p95 622
  -> 431 ms (-31%), TPOT p95 119 -> 124 ms, requests per second 11.35 -> 11.03; and TTFT p95 589
  -> 419 ms (-29%), TPOT p95 116 -> 121 ms, requests per second 11.65 -> 11.34. In the first,
  prefill per request fell from 310 to 164 ms while each step grew from 117 to 122 ms. There, a
  256-token step after it moved TTFT p95 +5%; with the 256 rule, the second session's ladder
  stopped after async off.
- NVIDIA A10G, Qwen2.5-7B-Instruct, the KV pool by setup: the cap was sized to 37 on the previous
  run's cold pool of 88,976 tokens (0.3.10's rule gave 29, the run's own warm pool would give 43).
  The run at 37 had 0 preemptions, KV peak 83%.
- Capacity (A10G, Qwen2.5-7B-Instruct in docker, limits TTFT p95 620 ms and TPOT p95 47 ms): 6 of
  8 steps were within the limits; the highest was 9.23 requests per second at 32 clients. The
  64-client step missed the TTFT limit by 4.5%.
- The floor (A10G, Phi-3.5-mini-instruct, `--goal tpot --max-req-s-loss 20`): lowering the cap
  lost 30% of requests per second; the next lowering moved to Could help and the revert came
  first.
- fp8 weights through Marlin (A10G, Phi-3.5-mini-instruct, two sessions): TPOT p95 40.7 ->
  31.9-32.0 ms, requests per second 2.02 -> 2.56, KV pool x1.26, broken answers 0 -> 0. On a
  tool-call workload (Qwen2.5-7B-Instruct), tool calls stayed 200 of 200 identical.
- The fp8 KV cache on 8.0-8.8 (A10G, Qwen2.5-7B-Instruct): it served 600 of 600 requests, but no
  answer matched bf16's (mean word overlap 0.34, 84% under half their words), and at least 53
  repeated themselves, against none with bf16.

Validated on a T4 (Qwen3-4B) and an A10G (Phi-3.5-mini-instruct, Qwen2.5-7B-Instruct), vLLM
0.30.0. The L4 (8.9) rows come from vLLM 0.30.0's source and were not run.

## 0.3.10 (2026-10-11)

Goals and limits:

- `tensward analyse --goal throughput|ttft|tpot` says which metric the run is for: requests per
  second, TTFT p95 or TPOT p95. Each playbook entry now says, per metric, whether it improves,
  worsens or does not change it, and why. Under a goal, changes that improve its metric lead; one
  that also worsens another metric says so first in its cost line; one that does not move it
  waits under Could help; one that worsens it is left out and named. Expected numbers stay
  throughput-only. Every `Try:` command repeats the goal (unless it is throughput) and the limits
  given.
- `--slo-ttft-p95 MS` declares a TTFT limit (with `--goal throughput`: throughput within it).
  Goodput and the `limits:` line appear only for limits you declared (the flags, or the
  configuration's `slos.ttft_p95_ms` and `slos.tpot_p95_ms`); no default limit is assumed. When a
  run already broke a declared TTFT limit, a change that raises TTFT waits under Could help under
  every goal.
- New entries for latency, each offered only where its signal fires: `small-step-budget` under
  `--goal ttft` sets `max-num-batched-tokens` to 512 and, from 512, to 256; its cost says the
  first burst of requests can wait longer. `stream-interval` (`--stream-interval 8`) when the API
  server's CPU crossed its limit. `async-scheduling` is tagged as raising TTFT (a request that
  arrives while a step runs is scheduled a step later), so under `--goal ttft` it is left out.
  `lower-concurrency` is also offered under `--goal tpot`, as a trade-off, when decode memory
  bandwidth or a saturated GPU holds TPOT at the cap. No full CUDA graph entry: vLLM 0.30.0
  already captures full graphs by default.

Several changes in a row:

- `analyse` reads the project's earlier runs. A change already measured is not suggested again
  ("tried in run R: ... against run P, kept"); when the last change lost on the goal by more than
  the noise threshold, reverting it comes first; a change that helped less than its printed
  expectation names the limit that took over.
- The first change carries a short plan (at most three steps). When the KV cache projected at the
  new cap reaches the warning level, the plan predicts KV capacity; otherwise it says which
  crossed limit still binds. Only a limit not crossed yet is a prediction the next run can test
  (in practice, KV capacity); the next run says confirmed, not confirmed, or "still binds (not a
  test)".
- When the run queues at its cap, `kv-sized-concurrency` offers the cap that fills 85% of the KV
  pool a cold start gives, sized in this workload's tokens, wherever that is above the next
  power-of-two cap or that raise does not fit; it then takes the raise's place in Try first.
  `more-kv-memory`'s command carries the cap the larger pool holds.
- Under `--goal ttft|tpot` the report shows the measured trade since the project's first run.
  When the last step gained less than a quarter of the requests per second it cost (the knee), a
  change that would go further the same way (a lower cap, a smaller step budget) moves to Could
  help with that trade as its cost. The knee holds for later runs until that flag changes again.
- Under `--goal ttft|tpot`, when no change that improves the goal is left, a change that leaves
  the goal's metric alone, answers a crossed limit and helps another metric moves to Try first
  ("does not change TPOT (...); helps throughput and TTFT") before the ladder stops. Changes that
  worsen the goal stay out; the knee and the TTFT-limit rule still apply.
- `raise-prefill-batch` moves to Could help under `throughput` and `ttft` when prompt processing
  ran at 80% or more of the engine's measured prompt rate and no run with a larger budget measured
  a higher rate.
- A change that removes a flag (such as `drop-speculation`) gets a command, and a Try first with
  nothing to run says why.
- Each suggestion with a command says how long testing it takes on your setup, from the run's own
  load and measure times; a lowered cap scales the measured window by the ratio of the caps.
- With nothing left, the report says why: a saturated GPU (and, when every request in flight was
  running, that the clients set the batch); how close prompt processing came to the measured
  prompt rate and what is left; the knee; or every change was tried, with the best run.

Diagnosis:

- The GPU's utilisation, power, SM clock and throttle reasons are sampled once a second
  (`metrics.json` `diagnosis.gpu_state`). A GPU busy 95% or more and held at its power or thermal
  limit over the measured window is named saturated (`gpu_saturated`), with what can still help.
- Rules decide on the inputs they have and name the missing ones; "can't tell" is left for a
  missing input that would decide.
- KV room is stated in this workload's requests (at its longest and at its mean). A cold compile
  cache is flagged, with the warm figure when a warm run of the same settings exists.
- The hardware-ceiling line uses the tensor peak at the measured SM clock (memory bandwidth is not
  scaled).

Workloads:

- Tool-call conversations (assistant `tool_calls`, `role: tool` results) are accepted. An
  assistant turn with `content: null` is sent with empty text, and tool-call arguments must be
  JSON text. The prompt-capacity probe refuses a conversation that does not end with a user turn,
  and the report says so.
- No 10,000-record limit: a larger file is sent as a fixed random sample of what the run needs,
  with the seed recorded in `run.json`.

Measured with vLLM 0.30.0, one run per step unless stated; each figure holds for its own setup
only, and answer quality was not measured (answers differ between runs, with no noise floor):

- Tesla T4 at its 70 W cap, Qwen3-4B served as float16, short Q&A prompts, 128 closed-loop
  clients. The GPU was named saturated at its power limit, and the throughput stop said the
  clients set the batch. Under `--goal ttft` the report offered 512, then 256: TTFT p95 1024 ->
  577 -> 539 ms, -47% against the warm default, with requests per second +2% and TPOT p95 -2%.
  A second T4 session (two runs per arm) measured 256 against 512 at -7.3% and -6.9% TTFT p95,
  and turning async scheduling off at 512 at -29% TTFT p95 (TPOT p95 +3%, requests per second
  -2.5%), which sets the `async-scheduling` tag. The first burst of 128 requests got slower at
  each step (TTFT p95 1902 -> 2259 -> 3167 ms).
- NVIDIA A10G, Qwen2.5-7B-Instruct, starting at a cap of 8 with 128 clients under `--goal
  throughput`: `kv-sized-concurrency` to 29 gave +21% requests per second with 0 preemptions (KV
  peak 72%); two more steps moved it by under 1%, then the stop named 85% of the measured prompt
  rate. TPOT p95 rose from 94 to 381 ms over the ladder.
- The same model from a cap of 16 with 64 clients under `--goal ttft`: TTFT p95 -35% over two
  steps (caps 29 and 32), requests per second +7%, TPOT p95 +89%, 0 preemptions.
- Mistral-7B-Instruct-v0.3 from a cap of 32 with 64 clients under `--goal tpot`: TPOT p95 46.3 ->
  40.5 -> 36.3 ms at caps 16 and 8, requests per second -68%, TTFT p95 about 5x. The second step
  (TPOT -10% for requests per second -44%) was the knee, and the next run, after an unrelated
  change, did not offer lowering the cap again.
- Tool-call workload: 400 of 400 requests served with the tool-call parser on. A 12,000-record
  file: two runs sent the same 600-record sample.
- Test-time estimates were within 25% on 6 of 7 A10G steps; one took 39% less than estimated.

Known limits:

- No tested prediction has been confirmed on a GPU yet. On the A10G ladders, 6 of 8 checks were
  "still binds (not a test)" and 2 were not confirmed: a larger step budget, after which prefill
  was predicted to stall decode (it moved requests per second by -0.5%), and n-gram speculation,
  which the decode ceiling does not model.
- Reverting a loss has not run on a GPU (no step lost on its goal); it is covered by replays.
- The knee's quarter and the saturation thresholds are not calibrated. The knee came from one
  ladder, at 0.235 against 0.25, so a repeat may find it a step later.
- The KV pool depends on the compile cache: the same arguments gave 88,976 and 103,536 tokens cold
  and warm, so a sized cap and its projection can be off (projected 84%, measured 72%; projected
  62%, measured 72%).
- Smaller step budgets slow the first burst of requests.
- Turning async scheduling off is not offered yet.
- The goal-neutral promotion, the `raise-prefill-batch` move near the prompt rate and the command
  for a flag removal were checked on replays of these runs, not on new ones.

## 0.3.9 (2026-10-09)

Diagnosis names the limit a run really hit:

- Prompt processing is judged by its measured share of the engine's time: prompt tokens computed
  per second against the engine's prompt capacity. The capacity comes from a short probe of the
  run's own prompts before the workload (about 24 s, up to 60 s), cached in the project per setup
  and recorded in `run.json`. See [Load-limited, step budget and prompt
  processing](docs/cli.md#load-limited-step-budget-and-prompt-processing).
- Load-limited: when the server runs everything the test sends with room to spare, the report
  says the test does not push the server and suggests raising its concurrency; no server change
  is prescribed.
- The per-step token budget is read from vLLM's start-up log and recorded in `run.json`. When
  steps are full while requests wait, the run gets a step-budget finding and
  `raise-prefill-batch`; n-gram speculation is not offered under it.
- Queueing is reported as a symptom of prompt processing or the step budget while the running
  requests stay below the concurrency cap. At the cap, it stays a finding.
- Tool calls: when at least 20% of answers are tool calls and the client's time to first token
  is 0.5 s or more above the engine's, both are shown and both causes are named (the tool-call
  parser holds output until a call is complete, and API-server load adds to it).
- `run.json` records whether the compile cache was warm, and `tensward compare` warns when two
  runs differ in it: a cold start can shrink the KV cache.
- Shared prompt text is compared in the order the engine renders it. A short differing span
  ahead of long shared text (such as a date line in the system message) gets the new
  `stable-prompt-prefix` suggestion.
- An offer made because a signal is near its threshold needs a 5% margin from it.
- After a change, the next `tensward analyse` shows the line the before run expected next to the
  measured result (`compare.json` records it as `expected`).
- Within a group, changes are ordered by their expected effect, and a change that may alter the
  answers never leads.

Expected effect: `prefix-caching` and `raise-prefill-batch` (raising `max-num-batched-tokens`) show
an "expected: throughput ..." line (a range such as "x2.1 to x5.3", or "at least xN" when it is
wide), on an NVIDIA A10G only. The number comes from a model of the run's own limits (what the
clients keep in flight, the concurrency cap, the room in the KV cache, the per-step token budget and
the prompt capacity) and the line names the limit that binds. `raise-concurrency` shows no number:
on one held-out step it printed +7% to +58% and throughput fell 6% (x0.94). See [Expected
effect](docs/cli.md#expected-effect).

Validated: the estimates and their scoring were fixed and sealed before any held-out data was
recorded, then checked on one A10G session (vLLM 0.30.0, Qwen2.5-7B-Instruct and
Mistral-7B-Instruct-v0.3, three runs per arm): three single-change chains and two ladders that
followed Tensward's own suggestions. `prefix-caching` measured x4.98 and x5.43 against a printed
x2.1 to x5.3 and x3.2 to x7.8. `raise-prefill-batch` measured +25% against a printed +8% to +24%
(above its printed high), on one step only. `raise-concurrency` failed one of three held-out steps.
The ladders gained x5.11 (two steps) and x1.47 (one step) end to end. A planned L4 session could not
run (no cloud capacity), so nothing is claimed beyond the A10G.
The release build then ran live on the A10G with both models: 94 of 96 checks passed; the two
that did not were one missing line in `tensward compare`, fixed and checked on that run's data.

## 0.3.8 (2026-10-08)

Expected effect of a change, and the ranking that follows from it:

- `raise-concurrency` (on a T4, L4 or A10G) and `prefix-caching` (on an A10G) can show the
  throughput they are expected to give this run, on a line after their reason:
  "expected: throughput at least x2.9 (this run: ...)". The range is new over old requests per
  second, said by its low end when it is wide; the bracket says what was read from the run. A
  possible loss is printed as one. See [Expected effect](docs/cli.md#expected-effect).
- No line is shown unless the run is a closed-loop workload on one GPU (no tensor or data
  parallelism) with at most 2% of requests failed and a steady window of 30 s or more, and uses
  no speculative decoding, no structured output, no forced tool call and no model whose cache
  holds linear-attention (Mamba) state. `raise-concurrency` shows no line on a prompt-heavy run
  (four or more prompt tokens computed per generated token, prefix-cache hits not counted) or
  when the run cannot tell. `prefix-caching` needs the engine's token ids of every prompt, no
  images and no routed experts; it replays the prefix cache over the run's own send order and
  KV-cache size, with the prompts as the engine renders them (chat template and tools included),
  and shows no line when under 10% of prompt tokens would come from the cache, or when the replay
  does not say.
- A change whose expected throughput is at least x1.5 at its low end moves to the front of Try
  first, under the bottleneck it relieves. Critical gates (such as fit and failed requests) stay
  above it, a change that may alter the answers or is offered only because its signal is near its
  threshold never moves, and nothing else changes order. Below x1.5 only the line is added;
  without an estimate the report is the same as in 0.3.7.
- `report.json` suggestion blocks gain `expected` (`metric`, `low`, `high`, `inputs`, `note`)
  when there is an estimate; the key is absent otherwise. `schema_version` stays "1".
- A run whose engine returned the prompts' token ids and that offers `prefix-caching` keeps
  `prompt_blocks.jsonl`: per prompt, its token count and chained hashes of its token ids in
  KV-cache-sized blocks, which cannot be turned back into text. `requests.jsonl`, `metrics.json`
  and `run.json` are unchanged.

Validated on two held-out before-and-after runs per change, on one NVIDIA A10G (vLLM 0.30.0,
Qwen2.5-3B-Instruct), with each estimate fixed before its check: a weak check. Measured ratio
against the printed low end: `prefix-caching` x4.13 against at least x3.11, and x3.73 against at
least x2.84; `raise-concurrency` x5.25 against at least x2.26, and x1.19 against at least x1.24.
The second was a prompt-heavy run that came in about 4% under, so prompt-heavy runs now show no
`raise-concurrency` number. That rule was set after the check, so the number shown rests on the
one other run. On the T4 and L4, `raise-concurrency` rests on earlier recorded runs only. The live
path was checked on the A10G with Qwen2.5-7B-Instruct and Qwen2.5-1.5B-Instruct.

## 0.3.7 (2026-10-08)

Engines are now adapters behind engine-neutral contracts. vLLM results are unchanged except for
the fixes below: the same diagnoses, suggestions and files, checked by replaying every recorded
run.

Changed (internal):

- An engine is an adapter that declares its capabilities (what it reports, with what quality,
  and which settings it can change), a signal source, a readiness probe and calibration
  profiles. The playbook and the calibration are stored as data, in `tensward/data/playbook.toml`
  and `tensward/data/calibration.toml`.
- New runtime dependency: `packaging>=22`.

Added:

- A serving configuration's `case` accepts `max_context_len`, `max_concurrent_requests`,
  `prefill_batch_tokens`, `kv_memory_fraction` and `prefix_caching` as names for `max_model_len`,
  `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization` and `prefix_cache`. A
  project's identity does not change with the spelling, and giving both names for one field is
  refused. These names need 0.3.7 or later (0.3.6 refuses them).
- Extension API 1 gains the names and keyword parameters listed in
  [`docs/extending.md`](docs/extending.md) under "Added within version 1 in 0.3.7". Earlier
  extensions keep working.

Fixed:

- With vLLM's data parallelism (`--data-parallel-size` above 1) every engine signal read as
  missing. Counters are now summed over the engines and kept out of one GPU's ceiling figures,
  per-engine gauges are never combined, and a data-parallel run reports no API-server CPU (one
  process's counter cannot cover several). Queueing, KV-cache capacity, prefill stalling decode,
  decode bandwidth, long context and API-server CPU say "can't tell: N data-parallel engines;
  per-engine limits are not modelled", and `cuda-graphs` is held back.
- With several API servers (`--api-server-count` above 1) their CPU counter is no longer named in
  the check that says vLLM may have renamed its metrics. The engine's step count is not read, since each API server counts only
  the steps that answered its own requests; the running batch comes from the client's timings
  (a few percent low, and the report says so), and the per-step figures say not measured, with
  the reason.
- A start-up wave that outlasted the run lost every polled peak, so a run under heavy KV
  pressure could miss KV-cache capacity. Such a run is now measured whole, with the engine's
  counters taken from a scrape before the run; rates over requests count the wave's requests
  too, and the report says so. Where nothing was measured the classes say can't tell, and a run
  with no class judged says "No change suggested: this run did not measure enough to judge"
  instead of "Nothing to change".
- Long context on a MoE model is judged against the experts a batch of that size reads per step,
  not the fewest experts one token reads, so a large batch no longer reads as a false critical
  long-context limit.
- `trim-max-context-len` is held back on vLLM: a shorter limit frees no KV cache there (on an
  A10G, 512 and 32768 gave the same pool to within 0.05%). It lowers only the KV cache one
  full-length request needs at start-up.
- A change that may alter the answers (such as `fp8-kv-cache`) is named only under "also:" in
  Try first, never as a numbered step.
- A calibration profile is taken only for the engine versions it names.
- `Trace.steps` (extension API) counts engine steps from the per-step annotation of vLLM's
  default model runner (0.29 and later); steps that scheduled no token are not counted.
- MoE backend lines that vLLM writes without quotes or with several words are read.
- A `run.json` written by a newer Tensward still loads for comparison, and settings with keys
  this version does not know are refused by name.

Validated on an NVIDIA A10G (vLLM 0.30), in two sessions:

- The same results as 0.3.6 for the same project on the same box: the same diagnosis and
  settings, with throughput within 0.15%.
- Under heavy KV pressure, with a start-up wave that outlasted the run: 55 preemptions over 192
  requests, named as KV-cache capacity.
- Engine step counts exact on both model runners (V1 and V2), with the steps that scheduled no
  token left out.
- The MoE kernels of an unquantized MoE model (OLMoE) read from its log; two API servers read
  as described above; API-server CPU at 0.93 cores at 112 requests/s, named as the limit.
- Data parallelism was checked on a two-engine metrics exposition built from a recorded
  single-engine scrape, not on a real multi-GPU run.

Known limits:

- On a MoE model with many experts at a moderate batch, long context may read clear when it is
  not.
- With several API servers the running batch comes from the client's timings.
- Validated on an A10G only.

## 0.3.6 (2026-10-07)

Structured output per record, and its quality in the report:

- A chat record may declare its own `response_format` (OpenAI's `json_schema` or `json_object`,
  sent to vLLM as `structured_outputs` for that record alone; `{"type": "text"}` is accepted and
  constrains nothing) and `chat_template_kwargs` (sent as is, also to `/tokenize`). Both are part
  of the workload identity only when present, so registered projects keep their identity. `init`
  refuses a record format together with a configuration-level `structured_output` (text
  included), and a record format with a `tool_choice` that names a tool or is `"required"`.
- `init` refuses a new project whose configuration declares `structured_output` while a record
  names a tool in `tool_choice` (vLLM answers such requests with HTTP 400). A project registered
  before 0.3.6 keeps loading.
- The run report has a "Structured output (quality, not performance)" section over the answers
  that must be JSON: the share of invalid JSON, of whitespace until the token limit and of
  answers stopped at the token limit, and the looping prompts. `metrics.json` and `report.json`
  keep it as `structured_answers`.
- `compact-json` under Could help "(from your answers)", when at least 5% of those answers ran
  on whitespace until the token limit: it sets `--structured-outputs-config` with
  `"disable_any_whitespace": true`, and `"backend": "xgrammar"` when the backend is unset or
  `auto`. The answers can change in content, not only spacing: a prompt the schema cannot hold
  may get a short, valid but meaningless JSON answer, and other answers may get longer.
- Invalid JSON is counted only for a `json` or `json_object` constraint and for records with a
  `response_format` that is not text. A configuration-level `regex`, `choice`, `grammar` or
  `structural_tag` constraint no longer counts its answers as invalid JSON.
- The "structured output with tools" check fires only for requests that carry both.
- `init` and `analyse` warn before anything runs when prompts offer tools the setup cannot serve
  (tool calling off, or no tool-call parser). The warning names both flags
  (`--enable-auto-tool-choice --tool-call-parser <parser>`); for a setup from `--current` it says
  to add them to that command and run `init` again.
- With `--current`, a configuration's `tool_calling: true` that the command does not enable is
  listed with the ignored serving fields.

Measurement:

- TTFT counts the first reasoning token: the client reads vLLM 0.30's `reasoning` delta (and
  `reasoning_content`, its earlier name). On a reasoning model TTFT is now the time to the first
  generated token, not to the first answer token, so it reads earlier than in 0.3.5. TPOT on a
  reasoning model now spans the thinking too, so it reads higher, and is now correct: the token
  count always included the reasoning tokens.

Hybrid (linear-attention) KV caches:

- `more-kv-memory` also raises `max-num-seqs` to the sequences the larger pool holds, when the
  cap held the running requests, and widens a pinned CUDA graph bound to match. At a share of
  0.95 or more it is listed under Not applicable here with the pool's numbers.
- `trim-max-context-len` is not offered on a hybrid cache (it adds no blocks), nor is
  `fp8-kv-cache` while the longest request fits in one attention block; each is listed under Not
  applicable here with the reason.
- `raise-concurrency` counts whole blocks: it offers the demand up to the requests the usable
  blocks hold, instead of a power of two halved until the projected usage fits. With 16 of 56
  blocks free at 4 per request, a cap of 10 now rises to 14, where 0.3.5 said the raise was
  blocked. Plain caches are unchanged.
- `kv_block_tokens` in `metrics.json`: the tokens per attention block, from the engine.
- The memory share the playbook assumes when the setup sets none is vLLM 0.30's 0.92 (was 0.9).
- A hybrid pool whose free blocks cannot hold one more request, while requests wait, is
  diagnosed as KV-cache capacity ("likely"). Two recorded runs of a hybrid model that read
  queueing before scheduling now read KV-cache capacity.

Validated on an NVIDIA A10G (vLLM 0.30):

- A 27B hybrid INT4 model with the new KV levers (memory share 0.95, the cap raised to the 7
  sequences the pool holds, CUDA graphs on): 1.23 requests/s and 141.9 output tokens/s, against
  0.80 and 92.6 with the CUDA-graph setup 0.3.4 suggested (+53%), with no failed requests and
  valid tool calls.
- A 7B AWQ model on structured chat, `compact-json` as suggested: requests/s 37.6 to 47.6
  (+27%), e2e p95 1881 to 1097 ms, invalid JSON 33% to 0%. The gain came mostly from the prompt
  the schema could not hold: it now ends at once with short, meaningless JSON instead of
  whitespace until the token limit. Decoding was not faster, and one healthy schema answer got
  about twice as long, so read the answer comparison before adopting it.

## 0.3.5 (2026-10-07)

Three changes can give a different diagnosis or suggestion than 0.3.4 on the same run:

- The average context per sequence in the hardware ceilings comes from each request's own prompt
  tokens, weighted by its time in the window, instead of the engine's prompt tokens divided by
  the requests. On workloads with long prompts the KV cache read per decode step is larger, and
  some runs that read "none found" now name attention / long context ("possible").
- When requests queue at the concurrency cap, the KV cache is too full for a higher cap, and the
  prompts share a prefix that prefix caching would hold once, `prefix-caching` is offered first
  for queueing, with the KV-cache use it projects at the higher cap, instead of a line saying the
  raise is blocked.
- What to try next follows the diagnosis. When the KV cache was full, a TPOT tail is put down to
  KV-cache pressure and `lower-prefill-batch` is no longer offered for it. `lower-concurrency` is
  offered only when preemptions, not the KV peak, decide the KV-capacity finding. The
  KV-capacity changes quote the diagnosis's evidence as their reason. Each finding in
  `metrics.json` has a `near` flag.

Also new:

- "API-server CPU: N cores" in the engine section: the CPU time of the engine's API-server process
  over the window. Near one core the API server limits throughput; the diagnosis names it as
  "API-server CPU" (0.7 and 0.9 cores, not yet calibrated, so at most "likely") and offers
  `more-api-servers`. With several API servers it is not judged.
- "prompt tokens per engine step" in the engine section and `prompt_tokens_per_step` in
  `metrics.json`. When a change raises the concurrency cap and TPOT p95 gets worse, "What changed"
  says why: most of each step is prompt processing, or more sequences share every step.
- `client_batch` in `metrics.json`: the sequences decoding at once, from the requests alone. The
  decode ceiling uses it when the engine does not count its steps.
- `report.json` beside `report.md`: the same report as data.
- `run.json` records the host: CPU model, physical and logical cores, RAM, swap and
  `vm.overcommit_memory`. `requests.jsonl` rows carry `prompt_tokens`.
- Each file of a run directory is written once; a directory without `metrics.json` is
  incomplete and never used as a baseline. Ctrl-C during the `--trace` or `--counters` launch
  still writes the run, with that section marked interrupted.
- The headline output rate uses the engine's token count instead of the requests' only on a
  steady window of 10 s or more.
- Extension API: new `tensward.api` (extension API 1), documented in
  [`docs/extending.md`](docs/extending.md). Extensions import only from it; one built for another
  API version is skipped with one line. The optimizer extension needs a release built for this
  API. `packages.json`, the format `serve start --from` reads, is documented there; one written
  by an earlier optimizer is still served.

## 0.3.4 (2026-10-06)

- `tensward --version` prints the version.
- With `--runtime local`, the vLLM version is read through the launcher's Python interpreter
  (`python -m vllm...` or a wrapper script) and, when the server log names it, from the log, so
  the report no longer says "version unknown" for these launchers.
- The workload in the serving configuration accepts `structured_output` (vLLM's
  `structured_outputs` object), `stop` and `logprobs` (sent as `logprobs` and `top_logprobs` on chat); they are part of the configuration
  identity. A schema or `chat_template_kwargs` per record is not supported yet.
- When the workload declares structured output, `ngram-speculation` is offered only under Could
  help, last, with a warning that drafts are often rejected under grammar-constrained decoding
  and a recommendation to compare answers with `--require-equal`.
- `structured_output` is validated: exactly one of `json`, `regex`, `choice`, `grammar`,
  `json_object`, `structural_tag` (plus its options). A wrong key such as `json_schema` is refused
  at `init`, naming `{"json": <schema>}`, instead of failing every request. A report check warns
  when prompts offer tools under structured output.
- `cuda-graphs` is offered alone only when the card has memory for graph capture beyond one
  full-length request; otherwise it comes with the change that frees the memory (language-model-only,
  a higher memory share, or a smaller context length), or is listed as not applicable.
- Qwen3.5-family checkpoints get the `qwen3_xml` tool parser. A server that fails because the host
  refused to map a large weight file now says to set `vm.overcommit_memory=1`, add swap or use a
  host with more RAM.
- The `cuda-graphs` suggestion bounds the capture size to the concurrency cap
  (`max_cudagraph_capture_size`) when it is known. On hybrid (linear-attention) models it also
  sets `max-num-seqs` to what the shared cache blocks allow, or is held back with the numbers.
  On a 27B hybrid INT4 model on an A10G that could not start with graphs before, the combined
  suggestion started and served 4.2x the output tokens per second of eager mode.
- When the KV cache is full and the time to first token is queue time, the diagnosis names KV
  capacity instead of prefill compute.

## 0.3.3 (2026-10-05)

- A model folder from a full download is accepted. The checkpoint is identified by the files the
  engine loads: the weight shards (named by the shard index, in any numbering), and the config,
  tokenizer and chat-template files. Everything else in the folder (documentation, licenses,
  notebooks, recipes, example scripts, subfolders) is ignored. A Hugging Face cache snapshot
  directory works too. Configs that point to custom code are accepted; that code runs only with
  `--trust-remote-code` in the registered setup (`--current`), and is then part of the model's
  identity; as an `--engine-arg` the flag is refused. This fixes refusals of full downloads of
  several popular repositories and of llm-compressor quantized ones.
- The report header and "What changed" show the same output rate; the no-noise-floor line appears
  once and counts the prompts that ran; answers that are all empty are reported as empty.
- What to try next has two tiers. Try first holds the changes for the diagnosed bottlenecks.
  Could help holds the rest, at most three in full, and adds near-threshold changes (a signal at
  0.7 of its gate or more, with the value and the gate in the reason). When raising
  `max_concurrent_requests` is blocked because the KV cache is too full, a line says so, and
  `more-kv-memory` is offered with its risk instead of "no change applies".
- A change whose reason reads your settings or prompts rather than a measured signal says so,
  for example "(from your prompts)".
- Prefill is counted from vLLM's prompt tokens by source, so prefix caching no longer reads
  0 tok/s of prefill.
- N-gram speculation is judged on the part of each prompt not shared with other prompts, so a
  long common system prompt no longer triggers it.
- `compare` and the "What changed" block: tool calls are compared per prompt and the report says
  for how many prompts they were identical; when the engine's and the requests' token counts
  disagree on a steady window, the engine's output rate is used and labelled; a comparison of
  two changed runs is titled "Run C compared with run B".
- Plainer wording for the no-bottleneck headline, the nothing-to-change line and the feedback
  line.
- When a calibrated class says "high" on a GPU other than the L4 and A10G it was calibrated on,
  the Diagnosis names those GPUs.

## 0.3.2 (2026-10-05)

- A Google Colab notebook, `examples/notebooks/tensward-colab.ipynb`, runs Tensward on a free
  T4 GPU in about 15 minutes: it measures a deliberately throttled setup, follows the first
  suggested change and prints what changed.
- Throughput is measured over a steady window that ends at the last declared dispatch, recorded
  in `metrics.json` as `window`. When the start-up wave outlasts the dispatches, or the run has
  no wave, the window is the whole run, measured as in 0.3.1. The old-baseline caveat now applies
  to any current-setup run recorded without a `window`, and a new caveat names the window kind
  of each run when the two differ.
- A window under 2 s is flagged, with the `request_count` that would give a usable one.
- The decode ceiling's average batch is counted by the engine (tokens per step) instead of
  sampled once a second, so short windows no longer change the diagnosis between identical runs.
  With speculative decoding on, decode memory bandwidth is "can't tell". The calibrated
  thresholds hold on a re-check of the calibration runs.
- The checks are printed in the terminal as `  check:` lines under the bottleneck, and are
  recorded in `metrics.json` as `checks`.
- When the engine cast a bfloat16 checkpoint to float16, the `Ran:` line ends with ", served as
  float16", the report says so, and `metrics.json` has `served_as_float16`.
- A fall of 90% or more shows as a factor such as ÷70 in "What changed" and the comparison table.
- `tensward env` warns when the installed torch-family packages were built for different CUDA
  versions, and `--json` has a `warnings` list.
- A `serve/state.json` that cannot be read is refused with a message naming the file; `serve
  start` used to overwrite it.
- Correction to the 0.3.1 throughput figures. 0.3.1 measured throughput over a span that
  included the start-up wave's tail and the drain after the last request. On the published runs
  its absolute throughput read 14%-27% low, and its gains were off by 4-10 points, in either
  direction. Recomputed from the recorded runs, over the steady window (output tok/s, 0.3.1 to
  recomputed):
  - Case 7, n-gram speculation: baseline 690.7 to 902.0, with speculation 616.8 to 841.6; the
    change is -6.7% (0.3.1 read -10.7%).
  - Case 8, Mistral-7B: 106.3 to 132.1.
  - Case 9, Gemma 4: 182.8 to 219.9.
  - Case 10, `max-num-seqs` 8 to 32: 298.6 to 362.5 and 754.1 to 878.5; the gain is +142.3%
    (0.3.1 read +152.6%).
  - Case 6, CUDA graphs on and off: both steady windows are under 2 s (0.69 s and 1.57 s), too
    short to measure, so there is no corrected figure or gain.

  0.3.2 measures client and engine numbers over one window when a run has a start-up wave.
  The case-study table and each affected report's header note carry the corrected figures.

## 0.3.1 (2026-10-04)

- A run made with `--engine-arg` opens with "What changed vs your current setup" (a run with
  `--baseline-answers` opens with "What changed vs your recorded answers"): the change,
  output and request throughput, TTFT p50 and p95 and TPOT p95 before and after (worse ones
  marked), the answers' verdict, the bottleneck before and after, caveats when the two runs may
  not be comparable, and a note to repeat both runs before trusting a difference of a few percent.
  The terminal prints it too.
- Closed-loop runs with `request_count` at least twice `concurrency` are measured in a
  steady-state window, after one wave of `concurrency` requests. The wave is reported on its own
  line and in `metrics.json` as `startup_wave`. Smaller runs include the wave and say so, and a
  comparison with a current-setup run recorded before 0.3.1 says that run included it.
- The classifier is calibrated on runs on an NVIDIA L4 and an A10G with vLLM 0.30: queueing and
  decode memory bandwidth can now say "high". The other classes, host overhead and speculation
  included, stay uncalibrated and top out at "likely", because their thresholds were not yet
  measured on runs that avoid the bottleneck. The diagnosis lists which classes were not crossed
  and which of those are uncalibrated.
- Speculation coverage, the share of generated tokens that came from accepted drafts, joins the
  tokens per draft in judging whether speculation pays. When it does not, `drop-speculation`
  is suggested.
- The quality line of the diagnosis follows the comparison's noise-aware verdict, so answers
  that differ no more than the current setup's own repeats no longer read as a problem.
- The notice before hashing model weights says how large they are and how long it takes.
- When vLLM's server log shows "JIT compilation during inference" inside the measured window,
  the report says so and suggests raising `warmup_requests`.
- `lower-concurrency` is suggested only when preemptions show it binds.
- The cost text of `kv-cache-dtype=fp8` is corrected: it may change answers, and it did not
  reliably change speed in controlled runs.
- The roadmap now lists CPU and offload with llama.cpp, edge devices, and multi-GPU hosts fifth.
- New case studies: four runs on an L4 (CUDA graphs on and off, n-gram speculation, Mistral-7B,
  Gemma 4) and the Qwen2.5-7B pair for the "What changed" block, in
  `examples/case-studies/`.

## 0.3.0 (2026-10-03)

- Python 3.11 is supported (3.12 was the minimum).
- `analyse` names the bottleneck that held the run back: queueing before scheduling,
  KV-cache capacity, prefill compute, prefill stalling decode, decode memory bandwidth, GPU
  compute, host overhead, long context, or speculation that does not pay. It gives the evidence,
  a confidence, which classes did not cross their thresholds and what this run could not tell.
  Fit, failed requests and answer quality are reported beside it.
- The thresholds are named in `thresholds.py`, each with what it means and where it comes from.
  None is calibrated on real GPUs yet, so no diagnosis says "high" confidence; calibration comes
  in 0.3.1. A run with fewer than 30 successful requests is capped at "possible".
- Suggestions come from a per-engine playbook and are grouped by the bottleneck they address,
  each with its evidence grade and what it may cost. A change ruled out for this model or setup
  says why. The report sections are now `## Diagnosis` and `## What to try next`; the latter
  replaces `## Suggested experiments`.
- `metrics.json` has a `diagnosis` object: every class's state and evidence, the thresholds used,
  and the changes that do not apply here.
- Official Mistral checkpoints register: `params.json`, `tokenizer.model.v3` and `tekken.json`
  are accepted, and a `consolidated.safetensors` copy is refused with how to remove it.

## 0.2.0 (2026-10-02)

Foundations for more engines, checkpoint formats and hardware. Nothing you measure changes.

- `tensward env` shows what Tensward detects on this machine: the GPUs with their driver and
  CUDA version, whether each engine is available as a docker image or a local command (and its
  version), the checkpoint formats it registers, and the engine, format and GPU combinations
  that work here. `--json` prints one object, for agents. `--engine`, `--runtime`, `--image` and
  `--local-command` check a specific setup. It starts nothing and writes nothing.
- `init` records the engine. It is the one your `--current` command runs, else `--engine`, else
  the first engine that serves the checkpoint on this machine (vLLM, for a safetensors
  checkpoint on NVIDIA). `init` prints the choice and why, and the JSON of `init` and `inspect`
  gains `"environment": {"platform", "engine", "engine_choice", "format"}`. Projects registered
  earlier are vLLM projects and keep their identities.
- `analyse` and `serve` use the project's engine. `--engine` on them is now a check: naming
  another engine than the project's is refused (`project_config_unsupported`).
- `analyse` and `serve` now refuse before starting when the engine isn't available (exit 2,
  `engine_unavailable`); before, the same setups failed while launching. The message says what
  is wrong (Docker missing, the daemon unreachable, the image absent, or the local command
  missing) and how to fix it. `init` only warns.
- Every `report.md` starts with one line saying what ran: the engine and its version, the image
  or command, the GPU with its driver and CUDA version, and the checkpoint format and
  quantization. `run.json` records `engine`, `engine_version`, `platform` and `format`. `analyse` prints the
  line on the terminal too.
- A directory of GGUF files is refused with a message saying GGUF comes with the llama.cpp
  engine (coming in a later release).
- With no GPU detected, the hardware ceilings and `--require-gpu` say "no supported accelerator
  detected".
- The last line of every `analyse` report, and of its terminal output, invites you to tell us
  about a finding worth sharing or a suggestion that was wrong, with the link to the issue
  tracker. It is plain text: nothing is sent.
- When a hardware ceiling cannot be shown, the headline says why ("hardware ceiling: not
  available (a share exceeded its bound: decode)") instead of "not measured".
- A local command given as a path (such as `~/venv/bin/vllm serve`) runs with that file's
  directory first on `PATH`, as if its environment were activated, so tools next to it (such as
  `ninja`) are found. When a server exits before it is ready, the failure names the first error
  line of its log that does not point elsewhere ("see root cause above").
- Fix: the decode ceiling for mixture-of-experts models is now a true upper bound (it assumes
  the fewest experts a step can read); it previously assumed uniform routing, which is the
  slowest case, so measured decode could exceed it.

## 0.1.6 (2026-10-01)

Output quality in every before/after: see whether a faster setup changed the answers.

- `analyse --engine-arg ...` compares the run with the newest run of your current setup: a
  verdict on the answers judged against that setup's own noise, the speed change, reference
  match and format health, three answer pairs in `report.md`, and a private
  `compare/<baseline>-vs-<candidate>/answers.md` with every prompt's answers side by side. The
  terminal prints the verdict. `--no-retain-responses` turns it off.
- `tensward compare --project P BASELINE_RUN CANDIDATE_RUN` does the same for any two runs of a
  project.
- Every `analyse` run writes `run.json`, recording what produced it. Runs from earlier versions
  cannot be compared.
- The suggestion that may change the outputs says that the report then compares the answers.
- `--require-equal` (on `analyse` and `compare`) exits with status 4 unless every answer is one
  the baseline gave for that prompt: the same text, tool calls and finish reason, over every
  successful answer. The report gives the first differing character per prompt and how often the
  current setup reproduced its own answers, with the batch-invariant mode as advice when it did
  not. `--baseline-answers FILE` compares with recorded production answers instead of a run.
- Every `analyse` run records each GPU's name, driver, PCI device id, VBIOS and SM count, and
  the newest CUDA the driver supports, in `run.json`. A comparison warns when the GPU or the
  driver differs. `--require-gpu NAME` and `--require-driver VERSION` refuse a mismatched machine
  before the model loads (`gpu_mismatch`).
- On machines with mixed GPU models, `--runtime local` now numbers GPUs as nvidia-smi does
  (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, unless you export it yourself).
- Fix: a suggested command replaces an override you already gave, instead of repeating the flag.

## 0.1.5 (2026-10-01)

Fixes from a first run on a real production command and checkpoint.

- Serving flags that count tokens, such as `--long-prefill-token-threshold 0`, are kept in the
  imported command instead of being treated as secrets. `HF_TOKEN`, `--hf-token` and `--api-key`
  are still blanked. `-cc` is read as `--compilation-config`, and dotted compilation keys keep
  their underscores. Re-running `init` with a command that has such a flag, `-cc` or a dotted
  compilation key now gives `project_inputs_changed`: register the project again.
- Suggestions that raise concurrency now raise a pinned `cudagraph_capture_sizes` (or its
  maximum) with it, so the larger batches keep their CUDA graphs. Other suggestions leave pinned
  sizes alone, and a report check says when running sequences exceeded what the graphs cover.
  The capture sizes are read from `--compilation-config` and `--max-cudagraph-capture-size`;
  `--cudagraph-capture-sizes` and the dotted capture-size keys are refused at import with the
  spelling to use.
- The n-gram suggestion turns async scheduling off, which vLLM v0.30 requires, and says so. It
  is listed as not applicable, with the reason, when pinned CUDA graph sizes cannot hold its
  steps, or when it would shrink the batch the graphs cover. A suggestion that the engine's
  rules reduce to the current setup is dropped.
- `--quantization auto_gptq` (and the other names vLLM resolves to the checkpoint's method) is
  accepted on a matching checkpoint; a different method is still refused.
- GPTQModel checkpoints register: `quant_log.csv` is ignored and the paths under
  `quantization_config.meta` are not treated as file references.
- The configuration accepts `open_loop` and `capped` arrivals, as the documentation says. Fit
  checks one sequence, or the declared cap, when the workload sets no concurrency.
- Fit leaves out the vision and audio weights when the text model alone is served
  (`--language-model-only`, or every media limit at 0).
- Not changed: the 4-characters-per-token estimate, which is wrong for non-Latin scripts.
  `stop`, `structured_output` and `logprobs` still cannot be set in the configuration.

## 0.1.4 (2026-10-01)

Image workloads (validated with Gemma 4 26B-A4B AWQ on vLLM v0.30, NVIDIA A10G).

- Chat prompts can carry images as local files: OpenAI `image_url` parts with a path relative to
  the prompts file. The rules:
  - formats are PNG, JPEG and WebP, checked from their headers;
  - animated images are refused;
  - the bounds are 20 MiB and 40 megapixels per image, 16 images per prompt and 1 GiB per
    workload;
  - remote and inline (`data:`) URLs are refused, with a message that says how to fix them.
- Images are hashed into the workload identity. `inspect` and `analyse` name an image that
  changed, and an image that changed after registration is never sent.
- Images are sent as `data:` URLs, built when each request is sent through a bounded cache.
  Measurements count image tokens and split requests with and without images.
- Fit counts images at their maximum token count. It also counts the larger batch the engine
  uses while media inputs are on: on an A10G it is within 1% of the measured KV capacity.
- New setting `media_limits` (`--limit-mm-per-prompt`). `no-media-encoders` is not suggested
  when prompts send images.
- A request whose image changed after registration is reported as such, not as "no HTTP
  response".
- New example: `examples/prompts-images.jsonl` with generated invoice, table, chart and dashboard
  images (`examples/images/make_images.py`), and `examples/config-images.json`.

## 0.1.3 (2026-10-01)

Mixture-of-experts and image+text models (validated with Gemma 4 26B-A4B AWQ on vLLM v0.30).

- Image+text and mixture-of-experts checkpoints register. `init` and `inspect` show the
  checkpoint's anatomy: components, attention layers, experts and input types.
- `init` and `inspect` estimate whether the model fits your GPU (`fit`) before anything starts.
  The estimate is advice, never a refusal, and memory used by other processes is reported
  separately.
- Hardware ceilings for mixture-of-experts models count the experts each decode step reads, and
  prefill uses the active parameters. Vision encoders and the embedding gather are no longer
  counted as decode traffic.
- `analyse` reports the engine's measured KV-cache capacity next to the estimate.
- New suggestion `no-media-encoders`: when no prompt sends images, serve only the text model
  (`--engine-arg language-model-only`). The batch-size suggestion no longer goes below what an
  image+text engine accepts.
- Gemma 4 tool calls use the engine's `gemma4` parser automatically. The MoE kernel backend
  appears in the report.
- `-tp` / `-pp` in `--current` are read as tensor and pipeline parallelism.
- Fixes:
  - A tensor that appears in two weight files is refused (its bytes were counted twice).
  - A precision mismatch between the configuration and the checkpoint now names what the
    checkpoint provides.

## 0.1.2 (2026-10-01)

- Fix: concurrent writes of the same state file (for example `serve stop` while `serve start`
  is still saving its state) could fail with "No such file or directory": every atomic write
  now uses its own temporary file.

## 0.1.1 (2026-10-01)

- `tensward --help` and the package description now say what the open-source package does:
  profile and diagnose (it does not tune automatically).
- Releases are published from GitHub Actions with PyPI trusted publishing.

## 0.1.0 (2026-10-01)

First public release: `init`, `inspect`, `analyse` (Docker or local runtime, `--trace`,
experimental `--counters`) and `serve`, with vLLM v0.30 as the first engine.
