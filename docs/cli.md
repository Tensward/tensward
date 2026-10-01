# Tensward CLI

The `tensward` command has four commands: `init` and `inspect` register and verify a project,
`analyse` measures it, `serve` runs your current setup. For a walkthrough see the
[README](../README.md).

```text
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

plus `current_setup` and `weights` (the precision and quantization the checkpoint provides).

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
| `2` | a refusal about the local environment (a project directory that is not private, a busy project, a checkpoint that changed while it was read, an unexpected I/O error); also argparse's usage error |
| `3` | a refusal about a declared input that does not meet its contract |
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
| `project_config_unsupported` | the configuration asks for what the checkpoint does not provide |
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
| `checkpoint_unsupported` | custom code, multimodal input, an unsupported quantization or a file reference |
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
model.safetensors, or model.safetensors.index.json plus exactly the shards it names
```

`README.md`, `LICENSE*`, `NOTICE*`, `.gitattributes`, `.git` and `.cache` are ignored. Any other
file, directory, symlink or special file is refused.

- `config.json` declares distinct architectures, a positive `max_position_embeddings`, and a
  `dtype`/`torch_dtype` of `bfloat16` or `float16`.
- Custom code (`auto_map`, `trust_remote_code`), multimodal keys such as `vision_config`, and
  file references (`*_file`, `*_path` settings) are refused wherever they appear in `config.json`
  and the two tokenizer documents. The one exception: a tokenizer document may name one of the
  checkpoint's own tokenizer files. Keys of the token maps in `tokenizer.json` are tokens, not
  settings.
- Quantization is read from `quantization_config` in `config.json`; `quantize_config.json` or
  `quant_config.json` may fill in bits and group size, but quantization declared only there is
  refused, because the engine would not detect it.
- Each safetensors header must be valid JSON with well-formed entries whose data ranges tile the
  file exactly. An unquantized checkpoint may only hold tensors of the dtype `config.json` declares. Weight bytes are hashed, not
  read as tensors; nothing is executed.
- Files are opened without following symlinks, and the directory is scanned before and after
  registration so an edit made meanwhile is refused as `checkpoint_changed`.
- The project directory may not be the checkpoint directory or inside it.

## Analyse

`analyse` starts the engine with the registered model and baseline settings, waits until it is
ready, runs the warmup, offers the workload, scrapes the engine's metrics before and after, and
always stops the server. It writes `<project>/runs/<run_id>/` (`report.md`, `metrics.json`,
`requests.jsonl`, the raw metrics scrapes, the server log: its last 10 MB) and lists suggested experiments.

- Progress goes to standard error as lines like `[   45s] measuring: 60/120 requests (50%)`:
  launching the engine (runtime, image), waiting for the model (a "still loading" line every
  15 s), warmup, measuring, the trace and counters launches, stopping. The results go to
  standard output.
- The terminal shows the headline numbers (current setup source, throughput, TTFT and TPOT p95,
  failures, share of the decode ceiling, GPU busy/idle with `--trace`), then the suggestions and
  the run directory; the full report is `report.md`.
- `--engine` names the serving engine (only `vllm` today). `--runtime docker` runs the engine's
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
