# Tensward CLI

The `tensward` command has five commands: `env` shows what Tensward detects on this machine,
`init` and `inspect` register and verify a project, `analyse` measures it, `serve` runs your
current setup. For a walkthrough see the
[README](../README.md).

```text
tensward env     [--json] [--engine E] [--runtime docker|local] [--image I] [--local-command CMD]
tensward init    --project <dir> --model <checkpoint-dir> --config <serving-json> --prompts <workload-jsonl>
                 [--engine vllm] [--current "<command>" | --current-file PATH]
tensward inspect --project <dir>
tensward analyse --project <dir> [--engine vllm] [--runtime docker|local] [--image I]
                 [--local-command CMD] [--gpus 0,1] [--engine-arg KEY=VALUE ...] [--trace] [--counters]
                 [--slo-ttft-ms N] [--slo-tpot-ms N] [--ready-timeout S]
tensward serve start|status|stop --project <dir> [--name N]
                 # start also takes --from, --package, the runtime options, --engine-arg KEY=VALUE,
                 # --host, --port, --ready-timeout S
```

Set up a checkout with `make sync` (`uv sync --locked`). No GPU, model or
network is needed to run the checks.

## Environment

`tensward env` shows what Tensward detects: the platform and its devices, whether each engine is
available as a docker image and as a local command (and its version), the checkpoint formats,
and the combinations that work here. It starts no engine and writes nothing. `--engine`,
`--runtime`, `--image` and `--local-command` narrow it to one engine, one runtime, or a specific
image or command (`--image` and `--local-command` name the image or command of the engine being
checked). Each probe times out after 10 s and then reads as not available.

```text
platform: NVIDIA GPU, 1 device(s)
  0: NVIDIA L4 (driver 595.91.07, CUDA 13.2), 22.5 GiB
engine vllm (vLLM):
  docker image vllm/vllm-openai:v0.30.0: available, version 0.30.0
  local command vllm serve: not available; install vLLM (`pip install vllm==0.30.0`), or use --runtime docker
formats: safetensors (hf-safetensors)
works here: vllm + safetensors on NVIDIA GPU (docker)
```

Without a supported GPU the first line is `platform: no supported accelerator detected` and
`works here` says `nothing yet`.

`--json` prints one object:

- `platform`: `null` when no supported accelerator is detected, otherwise `name`, `label` and
  `devices` (each with `index`, `uuid`, `name`, `driver`, `driver_cuda`, `pci_device_id`,
  `vbios`, `sm_count`, `total_bytes`, `used_bytes` and `description`; a field that cannot be
  read is `"unknown"`);
- `engines`: per engine `name`, `label`, `formats`, `platforms`, and a `docker` and a `local`
  check, each with `target` (the image or command), `available`, `version`, `reason` and
  `how_to_get`;
- `formats`: each registered format's `name` and `label`;
- `combinations`: the engine, format and platform that work here, with the `runtimes` that have
  the engine available.

`reason` is one of the following when the engine is not available, each with its own
`how_to_get`:

| `reason` | meaning and fix |
|---|---|
| `docker_missing` | Docker is not installed: install Docker |
| `docker_unreachable` | the daemon is stopped or access is denied (Docker's first error line is quoted): start the daemon or add your user to the docker group |
| `image_absent` | the image is not on this machine: `docker pull <image>` (Tensward never pulls images) |
| `command_missing` | the local command cannot be run: install the engine, or use `--runtime docker` |

## Register a project

`init` registers one local checkpoint directory, one serving configuration and one JSONL
workload as a **project**: a private directory (`0700`) holding `project.json` (`0600`). It only
reads and hashes the inputs (the model weights too: about 1-2 min for a 5 GB model, so a pause is expected; `inspect` re-hashes them and prints a progress line). It starts no engine, loads no model and uses no network.

The record stores where the inputs are and their identities, never their contents:

- `artifact_fingerprint`: SHA-256 over the checkpoint files, framed with their relative names;
- `config_digest` and `workload_digest`: digests of the parsed configuration and workload;
- `snapshot_id`: binds those three and the current setup.

`inspect` derives all of this again from the inputs and refuses if anything differs from the
record. `init` is idempotent: repeating it with the same inputs returns the existing record, and
it never rebinds a project to other inputs. Both commands print one line of JSON on standard
output (identities and the current setup, no input path, no prompt):

```json
{
  "artifact_fingerprint": "9f1c…",
  "config_digest": "2b7d…",
  "project_id": "0f2a…",
  "registration_state": "registered",
  "schema_version": "1",
  "snapshot_id": "6ac1…",
  "workload_digest": "c40e…"
}
```

plus `current_setup`, `weights` (the precision and quantization the checkpoint provides), and two
derived sections that are not part of any identity:

- `anatomy`: what the checkpoint is made of, read from its config and tensor headers. It lists
  bytes per component (text, embedding, output head, routed and shared experts, vision, audio),
  attention layers (full or sliding window), experts per token, input types and the maximum
  tokens per image. Figures it cannot derive are listed under `unavailable` with the reason.
- `environment`: where the project runs on this machine. `platform` is the detected platform
  (`nvidia`, or `null` when no supported accelerator is visible), `engine` the recorded engine,
  `engine_choice` why it serves the project ("from your --current command", "from --engine",
  or "chosen for a safetensors checkpoint on NVIDIA GPU", with any other engine that would also
  serve it), and `format` the checkpoint format (`hf-safetensors`).
- `fit`: whether the model fits the selected GPU. The verdict is `fits`, `tight`, `likely does
  not fit` or `not checked`, with the memory figures and the reason. It is judged on the GPU's
  total memory, and memory other processes use is reported separately. It is an estimate made
  without starting the engine, so it is advice and never a refusal. No GPU, or a multi-GPU
  setup, gives `not checked`.

`--current` is the command you run today (`vllm serve ...`, `python -m
vllm.entrypoints.openai.api_server ...` or `docker run ... <image> ...`). It is the baseline:
`analyse` measures it and every result is reported against it. Known flags become
engine-neutral settings, every other flag is kept verbatim, and host, port, API key and served
model name are dropped because Tensward owns them. The GPUs the command selects
(`--gpus device=1`, `CUDA_VISIBLE_DEVICES=1`) are recorded in the current setup and are the default of
`--gpus`, which `analyse` and `serve start` use to pick the devices (docker `--gpus "device=..."`,
or `CUDA_VISIBLE_DEVICES` with `--runtime local`). The hardware ceilings model one GPU: the selected
one (device 0 without a selection); with more than one selected they say "multi-GPU not modelled". Any model path is accepted (container mounts,
other hosts): the registered `--model` is what gets measured, and a note records when the names
differ. A different quantization than the registered checkpoint is refused, as is a compose file. With
`--current` the command wins: the configuration's serving fields (`max_num_seqs`,
`gpu_memory_utilization`, ...) are ignored, and `init` and every report list the ones that
differ from your command. The configuration's workload is always used. Without `--current` the
baseline is the serving fields of the configuration. The current setup is part of the project identity; to change it,
register a new project.

**Engine choice.** `init` records the engine: `--engine` if given, else the one `--current`
runs (`vllm serve`, `python -m vllm.entrypoints.openai.api_server`, or `docker run` of an image
named vllm), else the first engine, in preference order, that serves the checkpoint's format on
the detected platform. Without a GPU, the format decides. `init` prints the choice and why. A
`docker run` of another image is read as the chosen engine's command. An `--engine` that
conflicts with `--current` is refused (`project_inputs_invalid`), as is a command no engine can
read, and an engine that cannot serve the checkpoint's format on this platform
(`project_config_unsupported`). The engine is part of the current setup: projects registered
before 0.2.0 are vLLM projects, and to change the engine you register a new project.

**Availability.** `init` also checks that the engine is there, the way the project will run it
(the image or local command of your `--current` command; without one, either Docker or the local
command), and prints a warning, not a refusal, when it is not. `analyse` and `serve start` refuse
instead, before a run directory exists or anything starts, with `engine_unavailable` (exit 2)
and the reason's fix (see the table under [Environment](#environment)).

### Refusals and exit status

A refusal prints one line of JSON on standard error and exits with a status derived from its code:

```json
{"code": "project_inputs_invalid", "message": "the serving configuration is invalid: rounds: ..."}
```

Messages never contain a declared value or the content of a prompt. An unexpected I/O error
(`runner_failure`) names the cause and the path, e.g. `I/O error: Permission denied: <path>`.
A usage error (a missing option) is argparse's own message.

### Exit status and error formats

| status | meaning |
|---|---|
| `0` | success (`serve status`: the server is healthy) |
| `1` | a run or server failure: `analyse` or `serve` could not start or finish (a docker or engine error, an unreadable file, a malformed optimize result, no request succeeded) or `serve status` found the server not running or unhealthy |
| `2` | a refusal about the local environment (a project directory that is not private, a busy project, a checkpoint that changed while it was read, a GPU that is not the one required, an engine that is not available for the runtime (`engine_unavailable`), an unexpected I/O error); also argparse's usage error |
| `3` | a refusal about a declared input that does not meet its contract |
| `4` | `--require-equal`: the answers differ, or equality could not be shown (no baseline, a skipped comparison, incomplete coverage); a refusal or failure (1, 2 or 3) wins over it |
| `130` | interrupted (Ctrl-C or SIGTERM); a server this command started is stopped first |

There are two error formats, by command. `init` and `inspect` print the one-line JSON refusal
above on standard error (machine readable: stable `code`, exit `2` or `3`). `analyse` and `serve`
print a plain one-line cause, `tensward <command> failed: <cause>`, and exit `1`; their
preflight refusals (a project that is not registered, a changed input) use the JSON form.

### After a hard kill

Ctrl-C and SIGTERM stop what the command started, and a second signal during that cleanup is
ignored so it always completes. SIGKILL (or a power loss) cannot be handled. A Docker server
restarts on its own (`--restart unless-stopped`), so find and remove it with:

```text
docker ps --filter label=tensward.run
docker rm -f <container>
```

A detached local server is stopped with `tensward serve stop`, which also works while `serve start` is still loading.

| code | meaning |
|---|---|
| `project_inputs_invalid` | the configuration, workload or `--current` is missing or invalid |
| `project_inputs_changed` | an input differs from what the project was registered with |
| `project_config_unsupported` | the configuration asks for what the checkpoint does not provide, or `--require-equal` has nothing to compare or no answers to compare |
| `gpu_mismatch` | `--require-gpu` or `--require-driver` does not match the machine |
| `project_layout_invalid` | the project directory is inside the checkpoint, or collides with an input |
| `project_record_invalid` | `project.json` is unreadable or inconsistent with its own identity |
| `project_snapshot_mismatch` | the project is not the expected `snapshot_id` (library use) |
| `project_not_registered` | no project at this location |
| `project_state_unsafe` | the project directory or record is not private to you |
| `project_busy` | another registration holds the project's lock |
| `project_publication_failed` | `project.json` could not be written |
| `checkpoint_layout_invalid` | a required file is missing or a document is malformed |
| `checkpoint_inventory_unexpected` | the checkpoint has a file or directory outside the supported shape |
| `checkpoint_inventory_unsafe` | the checkpoint has a symlink or special file |
| `checkpoint_unsupported` | custom code, an unsupported quantization or a file reference |
| `checkpoint_precision_unsupported` | unquantized weights are not BF16 or FP16 (or do not match `config.json`), or an unsupported quantized format |
| `checkpoint_changed` | a checkpoint file changed while it was read |
| `runner_failure` | an unexpected I/O error |

## Serving configuration

One JSON object with these keys (`slos` and the keys listed below the example are optional). Unknown keys are refused.

```json
{
  "schema_version": "1",
  "case": {
    "weight_precision": "bf16",
    "activation_dtype": "bfloat16",
    "max_model_len": 2048,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 2048,
    "kv_cache_dtype": "auto",
    "gpu_memory_utilization": 0.8,
    "prefix_cache": false
  },
  "workload": {
    "output_tokens": 32,
    "request_count": 10,
    "request_timeout_s": 60.0,
    "arrival": {"kind": "closed_loop", "concurrency": 1},
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0
  },
  "warmup_requests": 3
}
```

- `engine_build` (default `unspecified`), `rounds` (default 1), `objective` (default
  `highest_throughput`) and `workload.mode` (default `same_text`) are identity only: nothing reads
  them, so leave them out. `engine_build` is a token naming the engine release you expect; it is
  not probed. A project that declares them keeps its identity.
- In `case` only `weight_precision` and `activation_dtype` are required; a field left out is
  unset and the engine decides. `tool_calling` (default `false`) lets the model answer with tool
  calls. The pair must be what the checkpoint provides (`bf16`/`bfloat16` or `fp16`/`float16`
  when unquantized, following its `torch_dtype`;
  `int4`, `int8` or `fp8` for a quantized checkpoint, shown as `weights`), and `max_model_len`
  may not exceed the checkpoint's `max_position_embeddings`; otherwise `init` refuses with
  `project_config_unsupported`.
- `workload.api` is `completions` (default) or `chat`; every workload record must match it.
  `arrival` is `{"kind": "closed_loop", "concurrency": N}` (N requests in flight),
  `{"kind": "open_loop", "rate_rps": R}` (R requests per second whatever the server does) or
  `{"kind": "capped", "rate_rps": R, "max_inflight": N}` (rate R, never more than N in flight).
  TTFT and TPOT are measured from the moment a request is sent. With `capped`, a request that
  the cap holds back waits on the client first, and that wait is not part of its TTFT: under
  overload the TTFT stays flat while the client-side queue grows, so read goodput and the
  request count beside it.
- `slos` maps names to positive numbers.
- Without `--current`, a `case` that declares no serving field makes the baseline the engine's
  defaults ("engine defaults (no current setup provided)"); declared fields are reported as
  "declared in config".

## Workload JSONL

One JSON object per line, UTF-8, at most 10,000 records and 64 MiB of prompts. A record has an `id`
(unique) and either a `prompt` or chat `messages`:

```json
{"id": "first", "prompt": "a plain prompt", "reference": "an expected answer", "labels": ["sample"]}
{"id": "second", "messages": [{"role": "user", "content": "a chat prompt"}]}
```

- A chat record ends with a user message and may add `tools` (OpenAI function definitions), a
  `tool_choice` and a `max_tokens` cap. A chat workload needs a chat template in the checkpoint
  (`chat_template` in `tokenizer_config.json` or a `chat_template.jinja` file).
- A chat message's `content` may be a list of parts instead of a string: `{"type": "text",
  "text": ...}` and `{"type": "image_url", "image_url": {"url": "images/a.png"}}`. Only user
  messages carry images, at most 16 per prompt. The url is a path relative to the prompts file,
  to a PNG, JPEG or WebP of at most 20 MiB and 40 megapixels (1 GiB for the workload). A remote
  or `data:` URL, an absolute path or `..` is refused: save the image next to the prompts file and
  give its relative path. The images' hashes are part of the workload identity.
- `reference` and `labels` are review metadata. Every field is part of the workload identity,
  and record order is significant.
- Blank lines, malformed JSON, duplicate keys, NaN, unknown fields and duplicate ids are refused.
- Requests cycle through the records in order until `request_count` is reached.

## Supported checkpoint

A text-only safetensors checkpoint, unquantized BF16 or FP16, or quantized with `awq` (4-bit), `gptq`,
`compressed-tensors` (one config group: W4A16, W8A8 int8 or FP8) or `fp8`:

```text
config.json, generation_config.json, tokenizer.json, tokenizer_config.json      required
chat_template.jinja, special_tokens_map.json, added_tokens.json, vocab.json,
merges.txt, tokenizer.model, quantize_config.json, quant_config.json             optional
processor_config.json, preprocessor_config.json                                optional
model.safetensors, or model.safetensors.index.json plus exactly the shards it names
```

`README.md`, `LICENSE*`, `NOTICE*`, `.gitattributes`, `.git`, `.cache` and `quant_log.csv` (a
quantizer's log) are ignored. Any other
file, directory, symlink or special file is refused.

- `config.json` declares distinct architectures, a positive `max_position_embeddings`, and a
  `dtype`/`torch_dtype` of `bfloat16` or `float16`.
- Custom code (`auto_map`, `trust_remote_code`) and file references (`*_file`, `*_path`
  settings) are refused wherever they appear in `config.json`, the two tokenizer documents and
  `processor_config.json` / `preprocessor_config.json`. The one exception: a tokenizer document may name one of the
  checkpoint's own tokenizer files. Keys of the token maps in `tokenizer.json` are tokens, not
  settings, and nothing below `quantization_config.meta` (a quantizer's own record, such as the
  paths it staged files in) is read as a file reference.
- Quantization is read from `quantization_config` in `config.json`; `quantize_config.json` or
  `quant_config.json` may fill in bits and group size, but quantization declared only there is
  refused, because the engine would not detect it.
- Each safetensors header must be valid JSON with well-formed entries whose data ranges tile the
  file exactly. An unquantized checkpoint may only hold tensors of the dtype `config.json` declares. Weight bytes are hashed, not
  read as tensors; nothing is executed.
- Files are opened without following symlinks, and the directory is scanned before and after
  registration so an edit made meanwhile is refused as `checkpoint_changed`.
- The project directory may not be the checkpoint directory or inside it.

A directory of `.gguf` files is refused (`checkpoint_inventory_unexpected`) with a message that
GGUF comes with the llama.cpp engine, coming in a later release.

## Analyse

`analyse` starts the engine with the registered model and baseline settings, waits until it is
ready, runs the warmup, offers the workload, scrapes the engine's metrics before and after, and
always stops the server. It writes `<project>/runs/<run_id>/` (`report.md`, `metrics.json`,
`run.json`, `requests.jsonl`, `responses.jsonl`, the raw metrics scrapes, the server log: its last
10 MB) and lists suggested experiments. With `--engine-arg` it also compares the answers with the
current setup's (see [Compare](#compare)); `--no-retain-responses` leaves out `responses.jsonl`
and so the comparison.

- Progress goes to standard error as lines like `[   45s] measuring: 60/120 requests (50%)`:
  launching the engine (runtime, image), waiting for the model (a "still loading" line every
  15 s), warmup, measuring, the trace and counters launches, stopping. The results go to
  standard output.
- The terminal shows the headline numbers (current setup source, throughput, TTFT and TPOT p95,
  failures, share of the decode ceiling, GPU busy/idle with `--trace`), then the suggestions and
  the run directory; the full report is `report.md`.
- `--engine` must name the project's engine (recorded by `init`; only `vllm` today); without it
  the project's engine runs. Another engine is refused (`project_config_unsupported`). `analyse`
  refuses an engine that is not available for the runtime (`engine_unavailable`, with how to get
  it) before a run directory exists.
- Every `report.md` starts with one line saying what ran: `Ran: <engine> <version> (<docker
  image I | local command C>) on <device (driver, CUDA)>; checkpoint: <format>, <quantization>`.
  It lists the selected GPUs (device 0 without a selection); for a local command it shows only
  the executable name. `analyse` prints the same line first on the terminal.
- The last line of every `analyse` report and of its terminal output asks you to report anything
  worth sharing, or a suggestion that was wrong, at the project's issue tracker. It is plain text:
  nothing is sent.
- `--runtime docker` runs the engine's
  pinned image (or `--image`), which must already be present locally; `--runtime local` runs
  `--local-command` (default: the engine's own command).
- Tuning is expressed in engine-neutral settings (`max_concurrent_requests`, `max_context_len`,
  `kv_memory_fraction`, `prefill_batch_tokens`, `prefix_caching`, `cuda_graphs`,
  `kv_cache_dtype`); the engine maps them to its flags. `--engine-arg KEY=VALUE` passes an
  engine flag or overrides one of those settings by its flag name (for vLLM `max-num-seqs=64`).
  The report and the terminal header then say "current setup + overrides (...)".
- Settings the current setup leaves unset run with the engine's own default. The report lists
  those defaults where the engine module declares them (for vLLM 0.30: prefix caching on,
  `max-num-batched-tokens` 2048 on GPUs under 70 GiB with chunked prefill on, `max-num-seqs`
  256, `gpu-memory-utilization` 0.92, async scheduling on, CUDA graphs on), and the prefix-cache
  hit rate the engine reported.
- `report.md` ends its measurements with a "Checks" section when something makes the run less
  trustworthy: engine metrics that were not exposed, or tool calling the server cannot serve.
  Prefix caching is suggested when at least 20% of the prompts share a prefix of 256 tokens (or
  a quarter of their length) with another prompt.
- Reported: total tokens per second (prompt plus output), requests per second, and goodput, the
  requests per second that succeeded and met a per-request SLO (`--slo-ttft-ms`, default 1000;
  `--slo-tpot-ms`, default 100).
- `--trace` adds a separate, short profiled launch and reports GPU busy/idle time. `--counters`
  (implies `--trace`) then profiles the trace's top 3 kernels by GPU time (distinct variants by
  full name, one slot kept for attention) with Nsight Compute (`--clock-control none`). Metrics
  are collected in groups that each fit one replay pass (occupancy, memory, tensor): multi-pass
  collection left GPU memory behind on a cloud GPU. It needs `ncu` on PATH (NCU 2025.3+ for CUDA 13)
  and `--runtime local`; the docker runtime reports "counters need ncu inside the image". First
  a no-model probe (a fp16 matmul, eager and inside a CUDA graph) must, per group, show every
  metric in one pass and return GPU memory to its cold level; a failing group is reported
  unavailable (`runs/<id>/counters/qualification.json`). Counter launches start the engine's
  own CUDA profiler and run under `ncu --profile-from-start off`; the workload slice is bracketed
  by the engine's start/stop profile requests, so ncu counts only workload kernels (not startup
  warm-up) and ends the server itself once its small launch count is reached. ncu writes its CSV
  only when it exits, so the server is stopped afterwards and ncu given time to flush. Each
  qualified group is one such launch, filtered by the kernels' bare function names; because the
  trace and ncu spell template names differently, each trace kernel is matched to exactly one of
  ncu's printed names (function base name plus template arguments), else it is reported as not
  seen. Profiled numbers are diagnostic, never a speed claim.

### Example output

Excerpts from real runs on an NVIDIA L4 (Qwen2.5-7B-Instruct-AWQ, vLLM v0.30.0, Docker, the
repository's `examples/config.json` and `examples/prompts.jsonl`). The full reports are
[`examples/report-l4.md`](../examples/report-l4.md) and
[`examples/report-l4-suggested.md`](../examples/report-l4-suggested.md).

```text
- measured: output 724.7 tok/s, total 1783.8 tok/s, requests 8.91 req/s, goodput 1.11 req/s, TTFT p95 2111 ms, TPOT p95 20.3 ms, 0 of 160 requests failed, hardware ceiling reached 79%

## Hardware ceilings (theoretical upper bounds, not targets)

- decode ceiling at the measured batch: 929 tok/s
- decode measured: 725 tok/s
- decode, share of its ceiling: 78.0%
- prefill ceiling: 9,272 tok/s
- prefill measured: 130 tok/s
- prefill, share of its ceiling: 1.4%

## Suggested experiments

- `raise-concurrency`: running requests hit max_concurrent_requests 16 with 16 waiting, and the highest sampled KV-cache usage is only 0.9%; at your declared load of 32 concurrent clients, highest sampled running plus waiting was 32, so 32 is worth trying (the load is what your configuration declares, not measured traffic). It usually cuts queueing (TTFT) but slows each token (TPOT) as more sequences share every step - measure both
  Try: `tensward analyse --project ~/projA --engine-arg max-num-seqs=32`
- `raise-prefill-batch`: TTFT p50 1674 ms is over 20x TPOT p50 20.1 ms with requests waiting (prefill-bound). Larger prefill chunks usually cut TTFT but can stall running decodes (TPOT p95) - measure both
  Try: `tensward analyse --project ~/projA --engine-arg max-num-batched-tokens=8192`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.
```

Counters (`--trace --counters`, first two of the three kernels, verbatim):

```text
| kernel | GPU time | launches | grid x block | resources | theoretical occ | duration us | achieved occ % | eligible warps | issue % | DRAM % | tensor % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `marlin::Marlin<1125899906910725l, 1125899906843648l, 1125899906910725l, 1125899906910725l, 256, 1, 8, 8, false, 4, 8, false>` | 62.7% | 753 | 58 x 256 (58 SMs) | 148 regs, 99 KiB smem | 16.7% (registers + shared memory) | 40 | 16.6 | 0.13 | 10.8 | 85.0 | 13.3 |
| `cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x1_tn_align8>` | 16.5% | 6 | 9,504 x 32 (58 SMs) | 128 regs, 8 KiB smem | 20.8% (shared memory) | 4,384 | 20.6 | 0.02 | 1.7 | 96.5 | 3.3 |
| `flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, false, false, false, false, true, false, true, false>` | 2.5% | 161 | 320 x 128 (58 SMs) | 252 regs, 80 KiB smem | 8.3% (shared memory) | 18 | 8.3 | 0.13 | 12.6 | 50.1 | 22.8 |
```

Following the first suggestion and measuring again, on the same machine and workload:

| NVIDIA L4, Qwen2.5-7B AWQ, 160 requests, 32 clients | baseline (`--max-num-seqs 16`) | `max-num-seqs=32` |
|---|---|---|
| requests/s | 8.91 | 14.75 |
| TTFT p95 | 2111 ms | 372 ms |
| TPOT p95 | 20.3 ms | 23.1 ms |

This is one honest example, not a promise: one run, one GPU, one workload.

### Counters

`--counters` is experimental and needs all of these:

- `--runtime local`, with the engine (vllm) and Tensward installed in the same Python
  environment: the probe and the engine both import torch.
- `ncu` (Nsight Compute CLI 2025.3 or newer) on `PATH`.
- Permission to read GPU performance counters: run as root, in a container started with
  `--cap-add SYS_ADMIN`, or set the driver option `NVreg_RestrictProfilingToAdminUsers=0`.
  Without it the qualification probe reports `ERR_NVGPUCTRPERM`.

A recipe that worked on an L4 and an A10G is [`examples/counters/Dockerfile`](../examples/counters/Dockerfile)
(vLLM 0.30.0, ncu 2025.3.1, Tensward built from the checkout):

```sh
docker build -f examples/counters/Dockerfile -t tensward-counters .
docker run --rm --gpus all --cap-add SYS_ADMIN --ipc=host \
  -v <models>:<models> -v <project>:<project> \
  -v <config dir>:<config dir> -v <prompts dir>:<prompts dir> tensward-counters \
  tensward analyse --project <project> --runtime local --trace --counters
```

A project stores the absolute paths of its model, configuration and prompts and checks them on
every command, so each input's directory must be mounted at the identical path
(`-v /path:/path`). A path that is missing says so and reminds you of this.

Run `sudo tensward ...` on a project you own and root may use it; the files it creates are
handed back to you.

## Compare

`tensward compare --project P BASELINE_RUN CANDIDATE_RUN` compares the answers of two runs of
the project, given as ids under `<project>/runs/`. With `--baseline-answers FILE` it takes only
the candidate and compares it with recorded production answers. `analyse --engine-arg ...` makes the same
comparison with the newest run of the current setup (no engine arguments, same registration, at
least one successful request) without being asked, and appends it to its report as "Compared
with your current setup". A comparison that cannot be made never fails `analyse`: the report
says why in one line.

Each `analyse` run writes `run.json`: the snapshot id, the engine arguments and settings that
ran, the `engine`, its `engine_version` (`null` when it could not be read), the `platform`
(`nvidia`, or `null`) and the checkpoint `format`, whether responses were kept, the temperature, the runtime, GPU indices and image, and the
`VLLM_*` variables the engine inherited from the shell (only under `--runtime local`; docker
passes the `-e` variables, which are in the settings). It also records each GPU in use
(`gpus_identity`: index, UUID, name, driver, PCI device id, VBIOS and SM count) and the newest
CUDA version the driver supports (`driver_cuda`; not the CUDA the engine was built with). A
comparison names a different runtime, image, GPU or driver in one line above the speed table. A
run without `run.json`, from before this feature, cannot be compared.

Per prompt, using only successful requests:

- **Similarity** is word-overlap F1 over lower-cased words, averaged over the pairs of a
  baseline answer and a candidate answer (at most 16 answers per prompt and side). Tool calls
  are compared as the tool name and its arguments parsed as JSON, so key order and spacing do
  not matter; a text answer against a tool call is 0.
- **The noise floor** is the same similarity between the baseline's own repeats of the prompt.
  A prompt is changed when its similarity to the baseline is more than 0.15 below the floor. A
  prompt the baseline answered but the candidate never did makes the verdict "changed".
- A prompt that ran once has no floor and is not judged; when none has one, the verdict says
  there is no noise floor and shows the agreement and the answers only.
- A reference match (word F1 against a prompt's `reference`), the rates of failed requests,
  answers cut short (`finish_reason` `length`), empty answers and invalid JSON (with
  `structured_output`) make the verdict "changed" when they get worse by more than 0.15
  (reference) or 5 percentage points.

At temperature above 0 with a seed, repeats of a prompt sample the same answer, so the floor is
tight and any engine change that alters sampling shows as changed answers.

### Equality gate

`--require-equal` (on `analyse` and `compare`) writes everything as usual, then exits with
status 4 unless the answers are equal. It is refused, before anything starts, with
`--no-retain-responses`, and on `analyse` without something to compare (`--engine-arg` or
`--baseline-answers`). A prompt is **identical** when every one of its successful candidate
answers is one of the baseline's answers for it: the same text, the same tool calls (arguments as
canonical JSON) and the same finish reason, over every answer, not a sample. The answers are
equal when every prompt is identical, none went unanswered, the failure rate did not rise and
the baseline covers every prompt the candidate answered. With no baseline, a skipped comparison
or nothing to judge, the gate fails too. The report names the first differing character (0-based)
of each prompt, and how often the baseline reproduced its own answers. That count is evidence
only: it does not decide the gate.

`--baseline-answers FILE` replaces the baseline run with recorded production answers, one JSON
object per line:

```
{"prompt_id": "weather", "text": "", "tool_calls": [{"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}], "finish_reason": "tool_calls"}
```

`finish_reason` and `tool_calls` are optional and not compared when absent. The file must hold
only successful production answers: Tensward does not filter rows, and refuses a `prompt_id`
that is not in the workload. The production requests must match the workload's messages, tools,
`max_tokens`, temperature and seed, or every prompt will differ for reasons the report cannot
name. There is no speed table and no failure-rate comparison; the report says how many prompts
the answers cover, and `--require-equal` fails unless every prompt the candidate answered is
covered and the candidate had no failed requests.

### GPU checks

`analyse --require-gpu NAME` and `--require-driver VERSION` check the machine before the model
loads and before a run directory exists, and refuse with `gpu_mismatch` (exit 2). With no
supported accelerator detected they refuse with "no supported accelerator detected" (the same
code). The checks are made through the detected platform (NVIDIA today). The names match
after removing a leading `NVIDIA ` and ignoring case, otherwise exactly (`A10G` does not match
`A10`); the driver matches exactly or as a dotted prefix (`580` and `580.95` match `580.95.05`).
The GPUs checked are the selected ones (`--gpus`, or the `--current` command's), by nvidia-smi
index or UUID (a `GPU-` UUID may be abbreviated); MIG devices are refused. With no selection,
every GPU on the machine must match. The refusal prints what was detected, for example `GPU 0 is
NVIDIA A10 with driver 580.95.05, but --require-gpu wants A10G`. `--runtime local` sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` for the engine unless you set it, so CUDA's GPU indices match
nvidia-smi's (vLLM warns about this itself on machines with mixed cards).

### Files

`compare/<baseline>-vs-<candidate>/` is created `0700`. It holds `answers.md` (every prompt, then
the distinct baseline and candidate answers, at most 3 each, trimmed to 4,000 characters; images
are named by their path in the workload) and `compare.json` (the machine-readable comparison),
both `0600`, because they hold generated text.

`compare` refuses with `project_inputs_changed` when the two runs, or either of them, were not
made from the project's current registration (checkpoint, configuration, workload and current
setup), checked from `run.json` before any answer is read. It refuses with
`project_inputs_invalid` when a run does not exist, has no `run.json`, did not keep its answers
(`--no-retain-responses`), or lacks `metrics.json`, `requests.jsonl` or `responses.jsonl`.

## Serve

`serve start --from current` (the default without an optimize result) runs the engine detached with your current setup (the registered
`--current` command, or the configuration's serving fields) so applications can call its
OpenAI-compatible API on `--host` (default `127.0.0.1`) and `--port` (default 8000). State and the
API key live under `<project>/serve/<name>/` (`--name`, default `default`; the served model name
is `tensward-<name>`). `serve start` waits until the model is loaded and answering (up to
`--ready-timeout`, default 900 s, printing progress lines on standard error), then exits; the
server keeps running until `serve stop`. The state is recorded as `starting` as soon as the
server is launched, so `serve status` and `serve stop` work while it loads. If `start` is
interrupted (Ctrl-C, SIGTERM), fails or times out, it stops the server it launched and marks the
state `failed`. `serve status` exits 0 when the server is healthy and 1 otherwise.

`--engine-arg KEY=VALUE` (repeatable, as in `analyse`) serves the chosen setup with a change,
for example your current setup plus a recommended `max-num-seqs=64`. The state records the
overrides and `serve status` shows them.

`--from` is `current`, `latest` (the newest packaged result under `<project>/optimize/`, which
only the separate optimizer writes) or the id of one such result. By default `start` serves
`latest` if the project has one, otherwise `current`, and prints which one it chose.
