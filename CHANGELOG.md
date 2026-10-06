# Changelog

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
