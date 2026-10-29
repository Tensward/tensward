# Tensward CLI

The `tensward` command has these commands: `env` shows what Tensward detects on this machine,
`init` and `inspect` register and verify a project, `analyse` measures it, `compare` compares the
answers of two runs, `set`, `reject` and `report` keep, drop and re-read the changes `analyse`
suggests, and `serve` runs your current setup. For a walkthrough see the
[README](../README.md).

```text
tensward --version
tensward env     [--json] [--engine E] [--runtime docker|local] [--image I] [--local-command CMD]
tensward init    --project <dir> --model <checkpoint-dir> --config <serving-json> --prompts <workload-jsonl>
                 [--engine vllm] [--current "<command>" | --current-file PATH]
tensward inspect --project <dir>
tensward analyse --project <dir> [--engine vllm] [--runtime docker|local] [--image I]
                 [--local-command CMD] [--gpus 0,1] [--engine-arg KEY=VALUE ...] [--trace] [--counters]
                 [--goal throughput|ttft|tpot] [--slo-ttft-p95 MS] [--slo-tpot-p95 MS]
                 [--max-req-s-loss PCT] [--capacity] [--baseline-answers FILE] [--ready-timeout S]
                 [--window auto|full|SECONDS] [--remeasure | --reuse-run RUN] [--adopt-within-noise]
tensward import <run-dir> --project <dir>
tensward set     <run|last|registered> --project <dir> [--accept-answer-changes]
tensward reject  <run|last> --project <dir> [--reason TEXT] | --clear <run> | --list
tensward report  <run|last> --project <dir>
tensward serve start|status|stop --project <dir> [--name N]
                 # start also takes --from, --package, the runtime options, --engine-arg KEY=VALUE,
                 # --host, --port, --ready-timeout S
```

`tensward --version` prints `tensward <version>`. With `--runtime local`, the engine version in
the report's `Ran:` line is the one the launcher's Python interpreter reports (a `vllm` script or
`python -m vllm...`); when the server log names the version it started, that one is shown.

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

When the installed torch-family packages (torch, torchvision, torchaudio) were built for different
CUDA versions, which makes vLLM crash at start-up, a `warning:` line after `works here` names the
builds and the fix.

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
  the engine available;
- `warnings`: a list of warning lines, empty when there are none (the torch/CUDA mismatch above).

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
`gpu_memory_utilization`, ...) and `tool_calling: true` are ignored, and `init` and every report
list the ones that differ from your command. The configuration's workload is always used. Without `--current` the
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

**Tool calling.** `init` and `analyse` warn before anything runs when prompts offer tools and the setup has
tool calling off or no known tool-call parser, with the fix: both flags vLLM needs
(`--enable-auto-tool-choice --tool-call-parser <parser>`), and `--engine-arg
enable-auto-tool-choice` for one analyse run. When the setup comes from `--current`, the fix is
to add the flags to that command and run `init` again with it as a new project: the command
overrides the configuration's `tool_calling`.

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
- `case` also takes engine-neutral names for five fields: `max_context_len` (`max_model_len`),
  `max_concurrent_requests` (`max_num_seqs`), `prefill_batch_tokens` (`max_num_batched_tokens`),
  `kv_memory_fraction` (`gpu_memory_utilization`) and `prefix_caching` (`prefix_cache`). Either
  name may be used, not both for one field; the project stores the name in brackets, so both
  spellings give the same project.
- `workload.api` is `completions` (default) or `chat`; every workload record must match it.
  `arrival` is `{"kind": "closed_loop", "concurrency": N}` (N requests in flight),
  `{"kind": "open_loop", "rate_rps": R}` (R requests per second whatever the server does) or
  `{"kind": "capped", "rate_rps": R, "max_inflight": N}` (rate R, never more than N in flight).
  TTFT and TPOT are measured from the moment a request is sent. TTFT ends at the first
  generated token, a reasoning token included (vLLM 0.30 streams reasoning as `reasoning`), so on
  a reasoning model it does not include the thinking time. With `capped`, a request that
  the cap holds back waits on the client first, and that wait is not part of its TTFT: under
  overload the TTFT stays flat while the client-side queue grows, so read goodput and the
  request count beside it.
- `workload.structured_output` is vLLM's `structured_outputs` object with exactly one of `json`
  (a schema), `regex`, `choice`, `grammar`, `json_object` or `structural_tag`, and optionally
  `disable_any_whitespace`, `disable_additional_properties` or `whitespace_pattern`; for example
  `{"json": {"type": "object", "properties": {...}}}` (not OpenAI's `response_format` or
  `json_schema`; `init` refuses them). It is sent with every request, and a report check warns when
  prompts also offer tools, since the grammar leaves no room for a tool call; `init` refuses a new
  project in which a record also names a tool in `tool_choice` (vLLM refuses the pair), and a
  project registered before 0.3.6 keeps loading. `workload.stop` is a list of stop
  sequences and `workload.logprobs` a number from 1 to 20 (sent as `logprobs: true` with `top_logprobs` on a chat workload). All three apply to every request and
  are part of the configuration identity. A chat record may declare its own `response_format`
  instead (see Workload JSONL); a configuration that declares `structured_output` refuses records
  that do. Invalid JSON is counted for a `json` or `json_object` constraint and for records whose
  `response_format` is not text.
- `slos` maps names to positive numbers. `ttft_p95_ms` and `tpot_p95_ms` are the latency limits
  you declare for `analyse` (see [Goals and declared limits](#goals-and-declared-limits)); a
  command-line value overrides them.
- Without `--current`, a `case` that declares no serving field makes the baseline the engine's
  defaults ("engine defaults (no current setup provided)"); declared fields are reported as
  "declared in config".

## Workload JSONL

One JSON object per line, UTF-8, at most 64 MiB. A workload of more than 10,000 records that holds
more than a run needs (its requests plus warm-up) is sent as a fixed random sample of that many
records; `run.json`'s `workload_sample` records the seed and the counts, and every run of the
project sends the same records. A smaller file is sent whole, in file order.
A record has an `id` (unique) and either a `prompt` or chat `messages`:

```json
{"id": "first", "prompt": "a plain prompt", "reference": "an expected answer", "labels": ["sample"]}
{"id": "second", "messages": [{"role": "user", "content": "a chat prompt"}]}
```

- A chat record ends with a user message or a tool result, and may add `tools` (OpenAI function definitions), a
  `tool_choice` and a `max_tokens` cap, its own `response_format` (OpenAI's `{"type":
  "json_schema", "json_schema": {"name": ..., "schema": {...}}}` or `{"type": "json_object"}`,
  sent to vLLM as `structured_outputs`; `{"type": "text"}` is accepted and constrains nothing)
  and `chat_template_kwargs` (sent as is, also to `/tokenize`, for example `{"enable_thinking":
  false}`). Both are part of the workload identity when present. A record with a format cannot
  also name a tool in `tool_choice` (vLLM refuses it) or set it to `"required"` (the answer is
  then a tool call, with no text to judge as JSON). A workload counts as structured for the
  n-gram suggestion when any record declares a format; the "structured output with tools" check
  fires only for records that carry both. A chat workload needs a chat template in the checkpoint
  (`chat_template` in `tokenizer_config.json` or a `chat_template.jinja` file).
- A conversation may include tool calls. An assistant message carries `tool_calls` (each with an
  `id`, `"type": "function"` and a `function` with a `name` and its `arguments` as JSON text,
  as OpenAI sends them; `content` may be empty), and a message with `"role": "tool"` answers one
  call by its `tool_call_id` (its `content` may be empty). A result that answers no earlier
  call, and arguments that are not JSON, are refused when the project is registered. The
  messages go through the model's chat template like any others, also for token counts and the
  prefix-cache check.

  ```json
  {"id": "agent-1", "messages": [{"role": "user", "content": "Do I need an umbrella in Rotterdam?"}, {"role": "assistant", "tool_calls": [{"id": "call00001", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rotterdam\"}"}}]}, {"role": "tool", "tool_call_id": "call00001", "content": "{\"rain_mm\": 4}"}], "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]}
  ```

  Two limits. The prompt-capacity probe (see
  [Load-limited, step budget and prompt processing](#load-limited-step-budget-and-prompt-processing))
  skips a workload with records that end on a tool result, so such runs have no prompt-work share;
  the report says so. And some chat templates check the call ids: Mistral's wants ids of 9 letters
  and digits, and the server answers HTTP 400 for others.
- A chat message's `content` may be a list of parts instead of a string: `{"type": "text",
  "text": ...}` and `{"type": "image_url", "image_url": {"url": "images/a.png"}}`. Only user
  messages carry images, at most 16 per prompt. The url is a path relative to the prompts file,
  to a PNG, JPEG or WebP of at most 20 MiB and 40 megapixels (1 GiB for the workload). A remote
  or `data:` URL, an absolute path or `..` is refused: save the image next to the prompts file and
  give its relative path. The images' hashes are part of the workload identity.
- `reference` and `labels` are review metadata. Every field is part of the workload identity,
  and record order is significant.
- Blank lines, malformed JSON, duplicate keys, NaN, unknown fields and duplicate ids are refused.
- Requests cycle through the records in order until `request_count` (at most 1,000,000) is reached.

## Supported checkpoint

A text-only safetensors checkpoint, unquantized BF16 or FP16, or quantized with `awq` (4-bit), `gptq`,
`compressed-tensors` (one config group: W4A16, W8A8 int8 or FP8) or `fp8`:

```text
config.json, generation_config.json, tokenizer.json, tokenizer_config.json      required
chat_template.jinja, special_tokens_map.json, added_tokens.json, vocab.json,
merges.txt, tokenizer.model, quantize_config.json, quant_config.json             optional
processor_config.json, preprocessor_config.json                                optional
model.safetensors, or model.safetensors.index.json plus the shards it names
```

`chat_template.json`, `video_preprocessor_config.json`, `sentence_bert_config.json`,
`hf_quant_config.json`, `params.json`, `tekken.json` and Mistral's `tokenizer[.mm].model.vN[mM]` files
are optional and, when present, hashed with the rest. A quantization declared only in
`hf_quant_config.json` (ModelOpt) is refused as not supported yet. The weights are the shards the shard index names (any file naming,
including 6-digit totals and 0-based ordinals), or `model.safetensors`.

The identity covers only those files. Everything else in the model directory is ignored:
documentation, licences, notebooks, recipes, images, every subdirectory, and any `*.safetensors`
the shard index does not name (such as Mistral's `consolidated.safetensors`). With
`--trust-remote-code` in the current setup (`--current ...`) every file in the model directory
(Python files included) is hashed with the checkpoint, and an `auto_map` value that points to
another repository (`org/repo--module.Class`) is refused. `--engine-arg trust-remote-code` is
refused: register the project again with it in `--current`.

Refused: a second `*.safetensors` beside `model.safetensors` when there is no shard index, a
`*.gguf`, and `*.bin`, `*.pt` or `*.pth` weights when there are no safetensors. A symlink to a file
that counts is refused unless it points inside its own Hugging Face cache repository
(`snapshots/<rev>/file -> ../../blobs/<hash>`); to get plain files, use
`hf download <repo> --local-dir <dir>`.

- `config.json` declares distinct architectures, a positive `max_position_embeddings`, and a
  `dtype`/`torch_dtype` of `bfloat16` or `float16`.
- `trust_remote_code: true` and file references (`*_file`, `*_path` settings) are refused
  wherever they appear in `config.json`, the two tokenizer documents and
  `processor_config.json` / `preprocessor_config.json`; `auto_map` is refused in all but
  `config.json` (the engine runs it only with `trust-remote-code`, see above). The one exception:
  a tokenizer document may name one of the checkpoint's own tokenizer files. Keys of the token maps in `tokenizer.json` are tokens, not
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
always stops the server. It writes `<project>/runs/<run_id>/` (`report.md`, `report.json`,
`metrics.json`, `run.json`, `requests.jsonl`, `responses.jsonl`, the raw metrics scrapes, the
server log: its last 10 MB; see [Run directory](#run-directory)), names the bottleneck and lists
what to try next. With `--engine-arg` it also compares the
answers with the current setup's (see [Compare](#compare)); `--no-retain-responses` leaves out
`responses.jsonl` and so the comparison.

- Progress goes to standard error as lines like `[   45s] measuring: 60/120 requests (50%)`:
  launching the engine (runtime, image), waiting for the model (a "still loading" line every
  15 s), warmup, measuring, the trace and counters launches, stopping. The results go to
  standard output.
- The terminal shows the headline numbers (current setup source, throughput, TTFT and TPOT p95,
  failures, share of the decode ceiling, GPU busy/idle with `--trace`), then the bottleneck line,
  the next steps and the run directory; the full report is `report.md`.
- `--engine` must name the project's engine (recorded by `init`; only `vllm` today); without it
  the project's engine runs. Another engine is refused (`project_config_unsupported`). `analyse`
  refuses an engine that is not available for the runtime (`engine_unavailable`, with how to get
  it) before a run directory exists.
- Every `report.md` starts with one line saying what ran: `Ran: <engine> <version> (<docker
  image I | local command C>) on <device (driver, CUDA)>; checkpoint: <format>, <quantization>`.
  It lists the selected GPUs (device 0 without a selection); for a local command it shows only
  the executable name. When the engine's server log says it cast a bfloat16 checkpoint to
  float16, the line ends with ", served as float16" and the report's first block adds "served
  as float16 (the checkpoint is bfloat16)"; `metrics.json` records it as `served_as_float16`. `analyse` prints the same line first on the terminal.
- After the bottleneck line the terminal prints each check of the run as an indented
  `  check: ...` line (the same text as the report's "Checks" section).
- In a notebook (Colab): notebooks have no Docker, so use `--runtime local`, with vLLM installed
  in the notebook's Python. The CLI never downloads a model; the notebook downloads it in its own
  cell. [`examples/notebooks/tensward-colab.ipynb`](../examples/notebooks/tensward-colab.ipynb)
  does all of this on a free T4 GPU.
- The last line of every `analyse` report and of its terminal output points to the project's issue
  tracker for questions and results to share. It is plain text:
  nothing is sent.
- `--runtime docker` runs the engine's
  pinned image (or `--image`), which must already be present locally; `--runtime local` runs
  `--local-command` (default: the engine's own command). `--local-command` is the whole server
  command, subcommand included, such as `'/path/to/vllm serve'`: Tensward appends the model and
  the engine's arguments. It is copied into the run's reports and `run.json` with the values of
  secret-named variables and flags (`HF_TOKEN=***`, `--api-key ***`) blanked; still, export
  secrets in the environment rather than typing them into it. When `serve start` fails because the server exited, the
  message ends with the path of the server's log, kept in the server's directory.
- With `--runtime local`, `TENSWARD_ENGINE_LAUNCHER` adds a wrapper before the server command,
  for example `TENSWARD_ENGINE_LAUNCHER='taskset -c 0-3'`. It is split into shell words (quoted
  arguments are supported), without running a shell. If its last word is `--`, as in
  `TENSWARD_ENGINE_LAUNCHER='/path/to/launcher --'`, Tensward inserts `--env NAME=VALUE`
  pairs immediately before that word for the API key, registered and per-run engine environment,
  startup-plan switch and GPU selection. Such a launcher must accept those pairs and pass
  them to its engine. Without a trailing `--`, the wrapper receives the usual inherited
  environment. Tensward stops the launcher's process group; wrappers that manage an engine
  elsewhere must forward signals and engine logs. `serve.json` records the expanded engine
  argv without the launcher; `run.json` records the runtime, settings and GPU selection.
  A profiler launch prefix, such as the one used for `--counters`, bypasses the launcher.
- Tuning is expressed in engine-neutral settings (`max_concurrent_requests`, `max_context_len`,
  `kv_memory_fraction`, `prefill_batch_tokens`, `prefix_caching`, `cuda_graphs`,
  `kv_cache_dtype`); the engine maps them to its flags. `--engine-arg KEY=VALUE` passes an
  engine flag or overrides one of those settings by its flag name (for vLLM `max-num-seqs=64`).
  The report then says "Your current setup + overrides (...)", and the terminal header "current
  setup (declared in config) + this run's overrides: ..." ("registered setup" while a setup is
  kept).
- Settings the current setup leaves unset run with the engine's own default. The report lists
  those defaults where the engine module declares them (for vLLM 0.30: prefix caching on,
  `max-num-batched-tokens` 2048 on GPUs under 70 GiB with chunked prefill on, `max-num-seqs`
  256, `gpu-memory-utilization` 0.92, async scheduling on, CUDA graphs on), and the prefix-cache
  hit rate the engine reported.
- `report.md` ends its measurements with a "Checks" section when something makes the run less
  trustworthy (the same lines are in `metrics.json` as `checks`):
  - engine metrics that were not exposed, or tool calling the server cannot serve (`init` and
    `analyse` warn before anything runs when prompts offer tools and the setup has tool calling
    off or no known tool-call parser, with the flag to add);
  - a measurement window under 2 s: "the measurement window was only N s; throughput is noisy;
    raise request_count to about M". M scales the requests dispatched inside the window to fill
    10 s and keeps the rest, rounded up to 10. The text adds "and includes the start-up wave"
    when the window includes it;
  - a steady window that no request streamed inside, so throughput is not measured;
  - a start-up wave that outlasted the last request, so throughput covers the whole run;
  - the engine's counters could not be read at the last request, so they cover the run to its
    end rather than the window;
  - the engine's generated tokens and the requests' differ by more than 10% over the window.
    On a steady window of 10 s or more the headline output rate is then the engine's count,
    labelled "(the engine's count, see Checks)"; on a shorter window it stays the requests'.
  Prefix caching is listed under what to try next when at least 20% of the prompts share a
  prefix of 256 tokens (or a quarter of their length) with another prompt.
- When some answers must be JSON (a `json` or `json_object` constraint in the configuration, or
  a record's own `response_format`), `report.md` has a "Structured output (quality, not
  performance)" section: the answers that must be JSON, the share that is invalid JSON, the share
  that ran on whitespace until the token limit (a grammar loop: stopped at the limit, at least
  80% whitespace), the share stopped at the token limit, and the looping prompts. `metrics.json`
  keeps it as `structured_answers`.
- Reported: total tokens per second (prompt plus output), requests per second, and, only when you
  declared a latency limit, goodput: the requests per second that succeeded and met every
  declared per-request limit (see [Goals and declared limits](#goals-and-declared-limits)).
- The report header says what the run is for (`goal: ...`) and, when limits are declared, which
  (`limits: ...`). `report.json` records both under `goal`: `goal`, `limits`, and, when the run
  made suggestions, `plan`, `check` and `stop`, plus `trade` under a latency goal once the project
  has an earlier run, and `floor` when `--max-req-s-loss` was given under a latency goal (see
  [What to try next](#what-to-try-next) and [Ladders](#ladders)). Each `plan` step has `change`,
  `args`, `limit`, `signal` and `tested`. `check` has `run`, `change`, `predicted`, `actual`,
  `confirmed` and `held`, and `cant_tell` when the predicted limit is decode memory bandwidth and
  the run speculates. `trade` has `since` (the trade base: the latest run of the starting
  settings) and, for the goal's metric, requests per second and the other latency, a `[base, now]`
  pair under the metric's name in `metrics.json`, and `wave` (`before_ms`, `after_ms`, `requests`,
  `base_requests`) when the start-up wave got slower. `floor` has `base`, `floor` (a share),
  `req_s` (`[base, now]`) and `crossed`. Each suggestion block gains `test` (the time to test it)
  and, for the first Try-first change, `plan` (its prediction lines). The key is additive: runs
  from before 0.3.10 have none and read as the throughput goal.
- While the run is measured, `tensward analyse` samples the GPU (see [GPU state](#gpu-state)) and
  records the workload sample it sent, if any. Other callers of the Python API do not sample
  the GPU or record the sample.
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

### Goals and declared limits

`--goal` says which metric the run is for; the default is `throughput`.

- `throughput`: requests per second. Changes are ranked as in 0.3.9, and expected numbers are
  shown where [Expected effect](#expected-effect) allows.
- `ttft`: time to first token (p95). Queueing, prompt processing, the per-step token budget,
  prefill stalling decode and tool-call output delay are the findings that count, and the report
  names which of them held the run back.
- `tpot`: time per output token (p95). Decode memory bandwidth, prefill stalling decode, host or
  CPU overhead between steps, API-server CPU and a saturated GPU are the findings that count.

Other findings still show, after those. The Diagnosis section opens with the goal's line ("For
TPOT: decode memory bandwidth (...)", or "no single limit was measured as binding"); when the
bottleneck of the run overall is not one that counts for the goal, its line says so: "Bottleneck:
queueing before scheduling (confidence: high), for the run overall; it does not hold back TPOT
(see the line above)". Under `ttft` and `tpot` no expected number is shown: the
estimates are throughput models. Every `Try:` command repeats the goal (unless it is throughput,
the default), the limits and the floor given on the command line, the runtime (`--runtime`, with
`--image` or `--local-command` when given) and `--gpus` when given, so the next run is made for
the same purpose on the same runtime and devices.

Each playbook entry has a direction for throughput, TTFT and TPOT (`improves`, `worsens` or
`none`) and the reason, as data in `playbook.toml`. Under a goal:

- changes that improve the goal's metric lead; one that also worsens another metric says so first
  in its cost line, as `TPOT worsens (...)`;
- a change that does not move the goal's metric goes to Could help, with "does not change TTFT
  (...); helps TPOT" in front of its cost. When no change that improves the goal is left in Try
  first, such a change moves back to Try first if a crossed limit called for it and it improves
  another metric (for example `more-kv-memory` under `tpot`: "does not change TPOT (fewer
  preemptions, but more sequences in each step); helps throughput and TTFT"), so the ladder does
  not stop while a change that costs the goal nothing is untried. Changes that worsen the goal
  stay out, low-bar offers are not promoted, a change that raises TTFT past a declared limit the
  run already broke is not promoted, and the knee still applies. If the promoted change then hurts
  the goal's metric, the revert comes first;
- a change that worsens the goal's metric is left out; the section ends with one line naming
  them ("left out because they worsen TTFT: ..."), except a change of a setting this run already
  changed from the current setup (turning async scheduling back on after a run that turned it
  off): So far shows that change;
- changes for a gate (such as fit and failed requests) and entries from extensions, which have no
  directions, keep their place.

Entries added for the latency goals, each offered only where its signal fires:

- `small-step-budget` sets `max-num-batched-tokens` to 512, and from 512 to 256 (at 256 or less
  it offers nothing). It is offered only under `--goal ttft`, when TTFT p95 is at least 5 times
  TPOT p50 (first tokens wait behind large mixed steps), the current budget is 512 or more, the
  longest prompt plus the peak number of running decodes fit in a step of the budget it offers
  (512, or 256 from 512), and neither queueing nor the step budget crossed. TTFT improves; TPOT
  is unchanged when every prompt fits one step; the start-up wave can wait longer. A budget below the concurrency cap, which vLLM refuses, is never proposed.
- `stream-interval` sets `--stream-interval 8` when API-server CPU crossed its limit. Tokens then
  reach the client in bursts of eight; TTFT is unchanged.
- `async-scheduling` is offered when the setup turns it off (and speculation is not on). It is
  tagged as worsening TTFT ("a request that arrives while a step runs is scheduled a step
  later"), as measured on a T4: under `--goal ttft` it is left out, and under the other goals its
  cost line says that a new request joins one step later.
- `lower-concurrency` also serves `--goal tpot`, as a trade-off: when the running requests reached
  the concurrency cap, nothing was preempted, and decode memory bandwidth or a saturated GPU
  holds TPOT, fewer sequences per step make each token sooner at a cost in throughput and TTFT.
- `no-async-scheduling` sets `--no-async-scheduling` under `--goal ttft`, when the engine
  took two decode steps or more to prefill a request (see
  [Async scheduling off](#async-scheduling-off)).
- `prefix-caching` improves TTFT as well as throughput.

**Limits are declared, never assumed.** `--slo-ttft-p95 MS` (also spelled `--slo-ttft-ms`) and
`--slo-tpot-p95 MS` (also spelled `--slo-tpot-ms`) declare a limit; so do `slos.ttft_p95_ms` and `slos.tpot_p95_ms` in the
configuration, which the command line overrides. Goodput, the share of requests meeting the
limits and the `limits:` header line appear only for a declared limit; with none declared there
is no goodput line. In `metrics.json`, `goodput` and `slo_attainment` then count every
successful request: read them as limits only when `report.json`'s `goal.limits` is not empty.
(Reports written by the separate optimizer print the limits it works to.)

**A floor on requests per second.** A latency goal can give throughput away one step at a time.
`--max-req-s-loss PCT` (a percentage below 100; no default; 0 sets no floor) sets how much it may
give: under `--goal ttft` or `--goal tpot`, measured runs are judged against the trade base (the
latest run of the project's current setup; see [Ladders](#ladders)). There is no expected
number for a latency goal, so only measurements count.

- While the run is within the floor, a Try-first change that lowers requests per second says
  where the floor is in its cost line: "requests per second may fall: your floor is 1.08 (20%
  below run R's 1.35), and this run made 1.22; ...". A change lowers requests per second when its throughput effect worsens, or
  when it lowers the concurrency cap (below the run's peak of running requests when no cap is set).
- Once the run's requests per second are more than PCT% below the trade base's, those changes
  move to Could help with the cost line opening "past your floor: requests per second 0.92, 31%
  below run R's 1.35 (your floor: 20%); ...". If the run's own change crossed the floor, going
  back to the latest run still within it comes first: "this run made 0.92 requests per second,
  31% below run R's 1.35, past your floor of 20%: go back to run S's settings first (the newest
  run within the floor)". The best run so far is never one below the floor.
  That run never carries a change you rejected, and its settings never differ from the registered
  setup's in a setting that can change answers (the weight quantization, the KV-cache precision,
  speculative decoding or compact JSON) unless they are the kept setup's.
- `report.json` keeps the comparison as `goal.floor`: `base`, `floor` (a share, 0.2 for 20%),
  `req_s` (`[base, now]`) and `crossed`. Every `Try:` command repeats the option. With
  `--goal throughput` it is ignored, with a note.
- `tensward set` keeps the floor the kept run declared (in `kept.json`): an `analyse` of the kept
  setup without `--max-req-s-loss` uses it, with a note, and its `Try:` commands repeat it.
  `--max-req-s-loss 0` runs without it; `tensward set` of that run drops the kept floor.

With `--goal throughput` and `--slo-ttft-p95`, the goal is throughput within that limit. If the
run's TTFT p95 is already over it, the changes that lower TTFT come first, and under every goal a
change that raises TTFT moves to Could help, its cost line opening "TTFT p95 is 2400 ms, past your
limit of 2000 ms, and this change raises it"; otherwise a change that may push TTFT p95 past the
limit says so in its cost line ("TTFT p95 is 900 ms; may push it past your limit of 2000 ms").

### Changes that can affect answers

Some changes make a run faster by changing what the model computes: a smaller weight format, an
8-bit KV cache, speculative decoding, compact JSON. Tensward offers them when a measured limit
calls for them, warns first, and shows the answers before and after. It never says which answers
are better; you decide, and keeping a change is your command (`tensward set`).

**What Tensward reads.** Before the first launch of an `analyse`, Tensward asks the installed
engine what it supports, without loading a model. It runs a short Python snippet where the engine
is installed (inside the image for `--runtime docker`, with `--pull=never` and no network; with
the launcher's interpreter for `--runtime local`; 120 s at most) and reads the engine's
quantization methods, each method's minimum compute capability and the KV-cache types. It
intersects them with the GPU's compute capability. A method the engine or the GPU cannot run is
never offered. bitsandbytes and GGUF are out-of-tree in vLLM 0.30.0: they are offered only when
their plugin package (`vllm-bnb-plugin`, `vllm-gguf-plugin`) is installed. EXL2 and EXL3 are not
served by vLLM and are not offered. A read that fails never stops a run: the report says in its
Quantization section that it could not read the support, and why, and offers no weight
quantization. The result is recorded in `run.json` as `answer_options` (`read`, `engine_version`,
`capability`, `on_the_fly`, `kv_cache`, `checkpoints`, `unjudged` and `plugins`, or `read: false`
with `why`). The Quantization section ends with a line such as "This engine and GPU (compute
capability 8.6) can run: fp8 weights made from this checkpoint at start (8-bit weights, 16-bit maths);
pre-quantized int8 W8A8, fp8, AWQ int4, GPTQ int4 or W4A16 checkpoints; no fp8 KV cache (offered
only from compute capability 8.9)."

For vLLM 0.30.0 the tables are:

| | T4 (7.5) | A10G (8.6) | L4 (8.9) |
|---|---|---|---|
| fp8 weights made from your 16-bit checkpoint at start (`quantization=fp8`) | not offered: vLLM 0.30.0 fails to start it on a T4 (measured) | weight-only (W8A16), with `linear-backend=marlin` (measured) | W8A8 (from vLLM 0.30.0's source, not run) |
| pre-quantized int8 W8A8, AWQ int4, GPTQ int4, W4A16 checkpoints | yes | yes | yes |
| pre-quantized fp8 checkpoint | not offered (same kernel) | weight-only (W8A16), with `linear-backend=marlin` (the kernel measured on fp8 made at start) | W8A8 (from vLLM 0.30.0's source, not run) |
| fp8 KV cache | no | not offered: it starts for 128-wide attention heads, but broke the answers of the model measured (Qwen2.5-7B) | yes (from vLLM 0.30.0's source, not run) |

vLLM 0.30.0 reports 7.5 as the fp8 floor but picks the CUTLASS fp8 kernel below compute
capability 10.0, and it fails in the profile run on a T4 (`cutlass_scaled_mm_sm75_epilogue`) and
on an A10G (`cutlass_scaled_mm_sm80_epilogue`), both measured. With `--linear-backend marlin`
vLLM picks its Marlin kernel, which serves fp8 weight-only on an A10G (measured). So on compute
capability 8.0-8.8 the `fp8-weights` command is `quantization=fp8` with `linear-backend=marlin`,
and a pre-quantized fp8 checkpoint is named with that argument; below 8.0 neither is offered
(Marlin fp8 on a T4 is not measured). On 8.0-8.8 vLLM 0.30.0 runs an fp8 KV cache only through
FlashInfer, which it picks only for 128-wide attention heads; other widths get Triton, which
refuses fp8 below 8.9. On an A10G, Phi-3.5-mini (head dim 96) did not start with it, and
Qwen2.5-7B (head dim 128) started and served, but none of its 600 answers matched bf16's: the
mean word overlap was 0.34 and 84% shared under half their words, many repeating themselves. So Tensward offers an fp8 KV cache only from 8.9. The A10G
column is what was measured there; other GPUs from 8.0 to 8.8 (the A100) follow it unmeasured.

"Weight-only" means the stored weights are 8-bit and the arithmetic is still 16-bit: memory and
decode bandwidth improve, prompt processing is no faster. A W8A8 format (int8 on all three GPUs,
fp8 on the L4) also runs prompt processing in 8-bit arithmetic. The L4 rows come from the source,
not from a run, and the report says so. vLLM 0.30.0 has no on-the-fly int4 and no on-the-fly
dense int8: those need a pre-quantized checkpoint.

**What is offered.** Each is an entry in What to try next, with the signal that fired:

- `fp8-weights` (`quantization=fp8`) and, with its plugin installed, `bnb-4bit-weights`
  (`quantization=bitsandbytes`), for a checkpoint that is not already quantized, when decode
  memory bandwidth is the diagnosed limit or the KV cache binds. They have a `Try:` command.
- `quantized-checkpoint`: the 8-bit formats the GPU runs first, then the 4-bit ones. Tensward
  downloads nothing and makes no command for it. Its cost line says to register that checkpoint
  with `tensward init --project <new> --model <its directory>` (same `--config` and `--prompts`)
  and then run `tensward analyse --project <new> --baseline-answers <this run>/responses.jsonl`
  to see its answers next to this run's. It is also offered when prompt processing is the limit
  and the GPU runs a W8A8 format.
- `fp8-kv-cache` only where the GPU runs it (see the table). Where it cannot, it is listed under
  Not applicable here with the reason: "with vLLM 0.30.0, an fp8 KV cache is offered only from
  compute capability 8.9 (this GPU: 8.6)".
  When the support could not be read, it is still left out below compute capability 8.9, read
  from the GPU.
- `ngram-speculation` and `compact-json` (see [What to try next](#what-to-try-next)).

Under a goal, the changes that cannot affect answers come first, then 8-bit ones, then 4-bit ones.
A group of Try-first changes that all can affect answers comes after the other groups, and the
first change the report shows is the first that cannot affect them. A change that can affect
answers is always shown in full, with its warning, whatever the group's limit of three.

**The warning before.** The line after such a change says, in words, that answers may change and
that the next report shows them:

- "(answers may change: the next report compares them with this run's)";
- for a 4-bit change, "(4-bit: answers change more often than with 8-bit; the next report
  compares them with this run's)";
- for a different checkpoint, which no command makes, "(answers may change: compare them as
  below)";
- when the run kept no answers, "(answers may change; this run kept no answers, so run it again
  without --no-retain-responses to see them before and after)".

**Answers before / after.** When a run's settings differ from the run it was made from (the
newest earlier run that sets only flags this run sets) in the weight quantization, the KV-cache
precision, speculation or compact JSON, the report adds `## Answers before / after`, built from
the same comparison as [Compare](#compare):

- how many prompts' answers differ and how many are identical, with the mean word overlap of the
  text answers ("word overlap 0.69 on average, where 1 is the same words") and, where the earlier
  run repeated its own answers, how many differ by more than those repeats do. When either run
  called tools, tool calls and text are counted apart: "Tool calls: 200 of 200 identical (name
  and arguments); text: 149 of 200 differ from run R's (word overlap 0.87 ...)";
- the answers that repeat themselves or are empty, before and after: "answers that repeat
  themselves or are empty: 0 of 600 before, 53 of 600 after". This count is a floor: an answer
  cut short before it repeats, or one word long, is not counted. The comparison's share of
  "answers that repeat themselves (empty ones not counted)" is a different measure;
- the share of valid JSON before and after (where answers must be JSON) and of failed requests;
- the least similar prompts, up to three, side by side (300 characters each, as code, without
  control characters or terminal escape sequences), under "The 3 least similar prompts:", or one
  identical prompt when none differ;
- "Every pair is in quality.json in this run's directory. Tensward does not judge which answers
  are better."

**Broken answers.** When the share of answers that repeat themselves or are empty rises by more
than 5 percentage points against the earlier run, the report says so under its header ("Warning:
answers look broken after this change: at least 53 of 600 answers repeat themselves or are empty,
against 0 of 600 in run R, and on average they share 15% of their words with that run's. ..."), the
section repeats it, the keep section says so before the `tensward set` command, and `tensward
set` of that run prints a warning line (when it refuses, the warning comes first). It never
blocks: keeping the run is your choice. In "What changed" the verdict reads "answers changed:
...".

`quality.json` holds `before` (the earlier run), `after` (this run), `changes` (the levers),
`summary` and `pairs`. `summary` has the counts the section prints: `prompts`, `identical`,
`differ`, `judged` (prompts with a noise floor), `changed`, `tool_calls` (`prompts`,
`identical`), `text` (`prompts`, `differ`, `similarity`), `unanswered` and `broken` (`before` and
`after` as `[broken answers, answers]`, and `rose`). Each pair has `prompt_id`, `before` and
`after` (the answers exactly as the engine returned them, each with its tool calls as
`name(arguments)` lines; print them with care), `identical` (the texts
are the same), `judged` (the earlier run repeated the prompt, so `changed` means anything),
`changed` (more different than the earlier run's own repeats; false when not judged),
`similarity` (word overlap, 1 for the same words) and `broken` (`before`, `after`: how many of the
prompt's answers repeat themselves or are empty). When either run kept no answers
(`--no-retain-responses`) the section says so and no file is written. When an earlier report
already showed these settings, the section points to that run and no file is written. With no
earlier run to compare with, there is no section.

**A recorded run as the baseline.** `--baseline-answers FILE` accepts a run's own
`responses.jsonl`: rows with no text, no tool calls and no finish reason are skipped, and the
`request_id` and `truncated` fields are ignored. That is how a different checkpoint is compared
with a run (see `quantized-checkpoint` above).

### Capacity at a limit

`tensward analyse --capacity` answers "how many requests per second does this setup hold within my
latency limits?" with a measurement, not an estimate. It needs at least one declared limit
(`--slo-ttft-p95`, `--slo-tpot-p95`, or `slos` in the configuration; see
[Goals and declared limits](#goals-and-declared-limits)) and a closed-loop workload.

On one launch of the setup (the kept setup, or the one `--engine-arg` gives) it measures steps of
closed-loop clients: the powers of two up to twice the registered clients, at most the largest
eight. A step offers at least 32 requests (four per client), after a start-up wave of one request
per client. The steps use consecutive slices of the workload's prompts, so a prompt is not sent
twice on the launch unless the workload is smaller than what the launch sends. A step is **within
your limits** when its TTFT p95 and TPOT p95 are within the declared ones and at most 1% of its
requests failed. The result is the highest requests per second among the steps within the limits:
that step's measured figure.

```text
capacity within TTFT p95 <= 2000 ms and TPOT p95 <= 60 ms, measured on one launch in 7 steps:
clients   req/s   TTFT p95   TPOT p95   failed   within
      1    0.90     310 ms      24 ms     0.0%   yes
      2    1.70     330 ms      25 ms     0.0%   yes
      4    3.20     380 ms      27 ms     0.0%   yes
      8    5.60     520 ms      33 ms     0.0%   yes
     16    8.10   1,100 ms      48 ms     0.0%   yes
     32    9.00   2,600 ms      71 ms     0.0%   no
     64    9.20   9,800 ms     130 ms     0.0%   no
highest within your limits: 8.10 requests per second at 16 clients
steps: /home/me/p/lp/capacity/20261011T120000Z-ab12/steps/ (each with its own report); summary: /home/me/p/lp/capacity/20261011T120000Z-ab12/capacity.json
```

(These numbers only show the format.) When the first step outside the limits missed only a
latency limit, by 10% or less, a line says the capacity lies between the two steps: "close to the
limit: 64 clients missed TTFT p95 by 4.5%, so capacity lies between 9.23 and 15.93 requests per
second". When the largest step is within the limits, the capacity is
"at least" its figure, because the limit was not found. When no step is within them, the output
says how the smallest step missed. When the launch sent more requests than the workload has
prompts, the output says how many requests reused a prompt and whether prefix caching was on,
since those requests' TTFT may read low.

The run writes `<project>/capacity/<id>/capacity.json` (`limits`, `engine_args`, `prompts`, `sent`,
`reused`, `prefix_caching` and the `steps`) and each step's files under `steps/<NNNN>-clients/`.
It writes nothing under `runs/`, so it does not change what the next `analyse` suggests. It
refuses, before any launch, with one sentence on stderr, when no limit is declared ("tensward
analyse --capacity: it needs a declared limit: ...; without one every step is within it.") and
with `--trace`, `--counters`, `--require-equal`, `--baseline-answers` or a `--goal` other than
throughput; and, as a JSON refusal like the project checks, when the arrival is not closed-loop.

### GPU state

From the start of the command, Tensward samples the GPU about once a second with `nvidia-smi`
(utilisation, power draw, enforced power limit, SM clock, maximum SM clock and the power and
thermal throttle reasons; no counters or profiler are needed), and judges the samples that fall
in the measured window. For the busiest GPU of the run, `GPU saturated` is critical only when
both hold:

- it was busy at least 95% of the window on average, and
- it was held at a power or thermal limit: the power-cap or thermal throttle reason was active
  in at least half of the samples, or, where the driver reports no reason, the mean power draw
  was at least 95% of the enforced limit.

Utilisation alone only means a kernel was running, so it never names the GPU saturated. When it
is saturated, the finding says that server settings cannot add throughput and names what can:
less work per request (shorter prompts or answers, cached shared prompts), a smaller or
quantized model, or a faster GPU. If the throttle reasons are not reported, the GPU is cleared
only when its SM clock averaged at least 95% of its maximum; otherwise the finding reads "can't
tell". Fewer than 5 samples in the window, no measured window, or no `nvidia-smi` sampling also
read "can't tell", with the reason; the run is never failed for it.

A saturated GPU also changes what else the report says: the load-generator check reads clear
("the GPU was saturated, so the test's load was enough"), the hardware ceiling is scaled by the
measured clock (see [Hardware ceilings](#hardware-ceilings-theoretical-upper-bounds-not-targets)),
and the advice is goal-aware (see [What to try next](#what-to-try-next)).

### KV room in this workload's requests

The report states the engine's KV-cache capacity in the requests of this workload, not in
maximum-length ones: "room for about 77 requests of this workload at its longest (N prompt + M
answer tokens), 120 at its mean (K tokens)". The longest is the longest measured prompt plus the
answer cap; the mean is the measured prompt plus output tokens per request. The line needs a
measured prompt length; without one it is left out. The "N full-length requests at once"
figure the engine reports is added when there is no such line or the longest request is at least
half the model's context length.

When the server started with a cold compile cache, the line says that this leaves less memory for
the KV cache than a warm restart (by up to a third), so the capacity may be smaller than a later
start holds. If an earlier run of the same resolved settings started warm, the line names it and
its capacity, and Tensward's estimate is compared against that warm figure. These are display
changes: the estimate itself is unchanged.

### Start-up wave and the measurement window

A closed-loop workload starts all its clients at once, so the first requests all wait for the
same prefill. When `request_count` is at least twice `concurrency`, `analyse` first offers one
wave of `concurrency` requests (continuing the prompt cycle past `request_count`, so the wave
warms no declared prompt's prefix unless the workload repeats prompts, as the shipped example
does with 160 requests over 10 prompts) and then the declared `request_count` requests. The wave
is reported on its own line (`start-up wave (first 32 requests, ...): TTFT p50 ..., p95 ...`)
and in `metrics.json` as `startup_wave`.

Throughput is measured over a window recorded in `metrics.json` as `window`, with `kind`,
`start_ns`, `end_ns` and `seconds`:

- `steady`: from the end of the wave to the last declared dispatch. A request that streams
  partly outside the window counts in proportion to the share of its streaming time inside it,
  for requests and tokens. The engine's counters are read over the same span. Latency
  percentiles cover every declared request.
- `whole_run`: from the first request to the end of the run, measured as Tensward 0.3.1 did. It
  is used when there is no room for a separate wave (fewer than twice `concurrency` requests;
  the report then says the wave is included in the numbers), for open-loop and capped arrival
  patterns, which have no wave, and when the wave was still running at the last dispatch (a
  check says so).

If the engine cannot be read at the last request, the engine's counters cover the run to its end
and a check says so. A steady window under 2 s is too short to trust and gets the short-window
check above.

A closed-loop run with enough requests also records its steady part in `window.json` as
`plan.steady`: the ramp until every client is in flight, the steady stretch, and the tail that
is left out. A KV-bound throughput step is measured on that shorter steady window after a bias
check, and under the ttft goal the report adds a line for the TTFT p95 of the middle 80% of the
window, marked as not judged. The number of requests a step sends is planned from its parent's
rate; a step that changes the answers sends at least 150 requests (the floor), so that there
are enough pairs to compare. `window.json` also
records `caches` (whether the compile cache was warm), `memory` (what the server logged about
its KV pool) and, for a step that was stopped, `startup_guard`.

A current-setup run recorded without a `window` (before 0.3.2) was measured over a different
span, and a comparison with it says so, naming the Tensward version that recorded it.

### Run directory

Each file in `<project>/runs/<run_id>/` is written once. The evidence (`requests.jsonl`,
`responses.jsonl`, the metrics scrapes `metrics_before.prom` and `metrics_after.prom`, and the
server log) is written as the launch stops. `metrics.json`, `report.json`, `report.md` and
`run.json` are written once at the end, in that order. A directory without `metrics.json` is
incomplete; it is never used as a baseline. When Ctrl-C stops a `--trace` or `--counters`
launch, the run is still written, and its trace or counters section reads "interrupted; the
measurement before it is complete".

- `requests.jsonl`: one row per request, with its timings, outcome, `prompt_id` and
  `prompt_tokens` (the server's count for that request, or the registered prompt's count when the
  response did not carry one), plus `cached_tokens` and `timings` when the server sends them.
- `prompt_blocks.jsonl`: written for every run whose engine returned every prompt's token ids and
  its KV-cache block size. One row per prompt: `prompt_id`, `tokens`, `block_tokens` and `blocks`,
  the chained hashes of the prompt's full KV-cache blocks as the engine renders it (chat template
  and tools included), so the expected effect of prefix caching can be recomputed from the run
  directory, by `tensward report` too. The hashes cannot be turned back into text.
- `run.json`: what ran (see [Compare](#compare)), including `host`, the machine beside its GPUs:
  `cpu_model`, `physical_cores`, `logical_cores`, `ram_bytes`, `swap_bytes` and
  `overcommit_memory` (Linux's `vm.overcommit_memory`: 0 heuristic, 1 always, 2 never). A value
  that cannot be read is `null`. `analyse` adds two keys: `answer_options` (what the engine and
  the GPU can run; see [Changes that can affect answers](#changes-that-can-affect-answers)) and
  `analysis` (the options its suggested commands repeat, the declared floor and the launch's
  timings), which `tensward report` reads.
- `quality.json`: written only when the run's answers are shown before and after a change that can
  affect them (see [Changes that can affect answers](#changes-that-can-affect-answers)).
- `report.json`: the report as data. `report.md` and the terminal output are rendered from it.
  It holds `schema_version` (`"1"`), `run_id`, `ran` (the `Ran:` line) and `sections`, in report
  order. Each section has an `id` (`header`, `what_changed`, `diagnosis`, `next_steps`, `setup`,
  `requests`, `performance`, `engine`, `defaults`, `ceilings`, `quantization`, `tool_calls`,
  `structured_answers`, `context`, `checks`, `trace`, `counters`, `extension`, `answers`,
  `answers_after`, `keep`, `feedback`), a `title`, a
  `title_audience` and `blocks`. Each block has a `kind` and an `audience` (`all`, `markdown`,
  `terminal`, or `json` for data only `report.json` holds):
  - `text`: `key`, `text`, `tone` (`plain`, `strong` or `warning`) and `item` (a list item);
  - `figure`: `key`, `label`, `value`, `unit`, `digits`, `grouped` and `note`;
  - `table`: `key`, `header` and `rows`;
  - `suggestion`: `name`, `tier` (`try_first` or `could_help`), `reason`, `near`, `basis`,
    `evidence`, `costs`, `quality_risk` and `command`; `test_seconds`, the estimated seconds
    from launch to the end of measurement, and `start_seconds`, the estimated seconds from
    launch to ready for the step's start class (each a number, or null when unknown); and
    `expected`, only when the change has
    an expected effect (see [Expected effect](#expected-effect)): `metric` (`"throughput"`),
    `low` and `high` (new over old requests per second), `inputs` (each with `key`, `value` and
    `text`) and `note`. The key is absent, not null, otherwise.

### What changed vs your current setup

A run made with `--engine-arg` after a run of the current setup (see [Compare](#compare)) opens
`report.md`, after the `Ran:` line, with `## What changed vs your current setup (run <id>)` ("the
registered setup" while a setup is kept); the terminal prints the same block. It shows:

- the change (`--engine-arg ...`);
- output tokens per second, requests per second, TTFT p50 and p95 and TPOT p95 (the run's goal
  metric first), each as before, after and change, with "within noise" added when the printed
  change is 2% or less (throughput) or 5% or less (latencies), so a printed "+2%" reads the same
  for every metric, "worse" when it moved further the wrong way (a rise
  in a latency, a fall in throughput), and "unchanged" when it moved by less than 0.5%. A fall of
  90% or more shows as a factor, such as ÷70, here and in the comparison table. Requests per
  second below 10 show two decimals;
- the answers' verdict, as in the comparison ("answers changed: ..." when they changed);
- the bottleneck before and after, when both runs were diagnosed ("unchanged (...)" when it is the
  same);
- when the change raised the concurrency cap and TPOT p95 got worse, a line saying why: "most
  of each step is prompt processing (about N prompt tokens per step)" when the prompt tokens per
  engine step are at least four times the average running batch, else "more sequences share
  every step";
- a closing note: "One run each: repeat both runs before trusting a difference of a few percent."

A run made with `--baseline-answers` opens with `## What changed vs your recorded answers` instead;
recorded answers carry no speed or diagnosis, so the block has the answers' verdict but no
speed or bottleneck lines.

Caveats are added under it when the numbers may not be comparable: the two runs were measured on
a different GPU, driver or engine version; the current setup's run has no recorded `window` (it
was measured by an older Tensward over a different span; the caveat names that version); or the
two runs were measured over different kinds of window (the caveat names each run's kind: steady
state or whole run); or the two runs started with a warm and a cold compile cache, which can
shrink the KV pool by up to a third. The full tables stay in "Compared with your current setup".

### Calibration, speculation and the quality line

Queueing (`queue_share`) and decode memory bandwidth (`decode_of_ceiling`) are calibrated on
runs on an NVIDIA L4 and an A10G with vLLM 0.30, so a diagnosis of one of them can say "high".
On other GPUs and engine versions the same thresholds apply, but they were not checked there:
when a calibrated class names the bottleneck with "high" confidence on a GPU other than those
two, the Diagnosis adds a line naming the GPUs it was calibrated on (the confidence stays
"high").
The other classes are not calibrated and top out at "likely". Host overhead (`gpu_idle`) and
speculation (`spec_coverage`) have not yet been measured on runs that avoid them. KV-cache
capacity and prefill did not meet the calibration bar (at least 90% of their induced runs named
correctly). Prefill stalling decode, GPU compute and long context were not induced.

The decode ceiling uses the average running batch counted by the engine over the window
(generated tokens, less accepted draft tokens, per engine step). With speculative decoding on,
decode memory bandwidth is "can't tell": the ceiling does not model speculation.

With the GPU sampled (see [GPU state](#gpu-state)), "% of ceiling" scales the datasheet's tensor
peaks by the measured SM clock against its maximum (memory bandwidth does not scale with the
clock), so a GPU held at its power limit is not shown far from a ceiling it cannot reach. The
headline then reads, for example, "hardware ceiling reached 33% (the datasheet peaks at the
measured SM clock, 1215 of 1590 MHz)", and the ceilings section adds the clock line. Without
the samples it shows the datasheet figure.

API-server CPU (`frontend_cpu`) is the CPU time of the engine's API-server process over the
window, in cores (the engine's `process_cpu_seconds_total`; Tensward's own once-a-second scrapes
are included). The API server is one process, so about one core is its ceiling. The class
crosses at 0.7 cores (warning) and 0.9 (critical). These thresholds are not calibrated, so it
tops out at "likely". With more than one API server the counter covers one process, and the
class is "can't tell". When it crosses, `more-api-servers` (`--api-server-count 2`) is offered.
The engine section shows the figure as "API-server CPU: N cores".

The engine section also shows the prompt tokens per engine step: the prompt tokens the GPU
computed (prefix-cache hits excluded) per engine step over the window. `metrics.json` keeps it as
`prompt_tokens_per_step`, and keeps `client_batch`: the sequences decoding at once, from the
requests alone (output tokens per second times the token-weighted time per output token). The
decode ceiling uses the engine's average running batch, and `client_batch` only when the engine
does not count its steps.

Speculation coverage is the share of generated tokens that came from accepted drafts. Together
with the tokens per draft it decides whether speculation pays: when either is below its
threshold the diagnosis is "speculation that does not pay", and `drop-speculation` is suggested.

The quality line of the diagnosis follows the comparison's noise-aware verdict:

- "answers match the current setup's": every answer is identical;
- "answers differ from the current setup's only as much as its own repeated answers do": the
  answers differ, but no more than the baseline's own repeats do;
- "answers differ from the current setup's (see the comparison below)": they differ more than
  that;
- "answers differ, and there is no noise floor to judge them against": the baseline has no
  repeated answers per prompt.

### Example output

Excerpts from real runs on an NVIDIA L4 (Qwen2.5-7B-Instruct-AWQ, vLLM v0.30.0, Docker, the
repository's `examples/config.json` and `examples/prompts.jsonl`). The full reports are
[`examples/report-l4.md`](../examples/report-l4.md) and
[`examples/report-l4-suggested.md`](../examples/report-l4-suggested.md) (0.2.0 output, before the
diagnosis section) and [`examples/report-l4-0.3.0.md`](../examples/report-l4-0.3.0.md) (0.3.0, with
the diagnosis).

The excerpt below was produced before 0.3.10, when goodput was counted against default limits
of 1000 ms TTFT and 100 ms TPOT. Now the goodput figure appears only when you declare a limit.

```text
- measured: output 724.7 tok/s, total 1783.8 tok/s, requests 8.91 req/s, goodput 1.11 req/s, TTFT p95 2111 ms, TPOT p95 20.3 ms, 0 of 160 requests failed, hardware ceiling reached 79%

## Hardware ceilings (theoretical upper bounds, not targets)

- decode ceiling at the measured batch: 929 tok/s
- decode measured: 725 tok/s
- decode, share of its ceiling: 78.0%
- prefill ceiling: 9,272 tok/s
- prefill measured: 130 tok/s
- prefill, share of its ceiling: 1.4%

```

The report then opens with `## Diagnosis` and ends with `## What to try next`. For example, from a real run:
Qwen2.5-7B-Instruct AWQ on an NVIDIA L4 with the shipped example workload (32 clients against a
concurrency cap of 8). The full report is
[`examples/report-l4-0.3.0.md`](../examples/report-l4-0.3.0.md).
That report is from 0.3.0, before the thresholds were calibrated. Queueing (`queue_share`) and
decode memory bandwidth (`decode_of_ceiling`) were calibrated on runs on an NVIDIA L4 and an A10G
with vLLM 0.30, so a diagnosis of one of them can say "high"; the other classes top out at
"likely". Diagnosed again with these
thresholds, the same measurements read "confidence: high" with no calibration note, and the
decode-bandwidth line reads "(high)" too. Since 0.3.5 the "not crossed" list also names API-server
CPU, as below. Since 0.3.6, on a hybrid (linear-attention) cache, requests waiting while the pool's
free blocks cannot hold one more request (at the blocks one running request held) count as
KV-cache capacity, at "likely".

```text
## Diagnosis

Bottleneck: queueing before scheduling (confidence: likely; thresholds not yet calibrated on real GPUs)
- evidence: requests spent 98% of their time to first token queued; 32 requests were in flight against a concurrency cap of 8
- also seen: decode memory bandwidth (likely): decode ran at 72% of the memory-bandwidth ceiling at the measured batch
- not crossed (uncalibrated thresholds): KV-cache capacity, prefill compute, prefill stalling decode, API-server CPU, attention / long context
- speculation: off or not reported
- can't tell here:
  - GPU compute, tensor-bound kernels — kernel counters: run with `--counters`
  - host / CPU overhead — a GPU trace: run with `--trace`
- not modelled: offload / PCIe (needs offload signals, which come with the llama.cpp engine); multi-GPU communication (Tensward measures one GPU)
- fit and failed requests: every request was served
- workload shape: 1.5 prompt tokens computed per generated token

## What to try next

Try first:
For queueing before scheduling:
- `raise-concurrency`: running requests hit max_concurrent_requests 8 with 24 waiting, and the highest sampled KV-cache usage is only 0.6%; at your declared load of 32 concurrent clients, up to 32 requests were in flight, so 32 is worth trying (the load is what your configuration declares, not measured traffic). It usually cuts queueing (TTFT) but slows each token (TPOT) as more sequences share every step - measure both
  evidence: strong; may cost: TPOT rises as more sequences share each step; more KV cache in use
  Try: `tensward analyse --project <project> --engine-arg max-num-seqs=32`
For decode memory bandwidth:
- no change in this engine's playbook applies here
Could help:
- `prefix-caching`: 2 of 10 prompts (20%) share a prompt prefix of up to 114 words with another prompt (from your prompts and settings)
  evidence: strong; may cost: a little GPU memory for the cache
  Try: `tensward analyse --project <project> --engine-arg enable-prefix-caching`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.
```

`## What to try next` has two tiers, ordered for the run's goal (see
[Goals and declared limits](#goals-and-declared-limits)). Try first holds the changes for the diagnosed bottlenecks,
grouped by bottleneck, at most three in full each. Could help holds the other changes the
measurements support, at most three in full, with the rest named on one `- also:` line. The first
`Try:` command in the section is the first Try-first step that has one, else the first
Could-help step that has one.

One entry is offered under Could help only: a signal at 0.7 of its gate or more, and at least 5%
below it, offers `raise-prefill-batch` (TTFT over TPOT, gate 20), with its value and the
gate in the reason, for example "TTFT p50 is 15x TPOT p50 with requests waiting; the gate is 20x".
These rank after the other Could-help entries. `lower-prefill-batch` is offered only once TPOT p95
is at least 2.0 times p50, never below its gate.
A larger step budget changes the chunk size, not the rate at which the engine computes prompts. So
under the throughput and TTFT goals `raise-prefill-batch` moves from Try first to Could help when
prompt work ran at 80% or more of the prompt rate the probe measured, unless an earlier run with a
larger budget measured a higher rate. Its cost line then opens "prompt processing already runs
near the prompt rate this engine measured, and a larger step budget changes the chunk size, not
that rate (no run here with a larger budget measured a higher one)". And
when queueing is diagnosed at the concurrency cap but the KV-cache usage is too high for a larger
cap to fit, the "For queueing" group says raising the cap is blocked, and `more-kv-memory` is
offered with that reason unless it is already set to its raised value or already listed. The usage
is "too high" when doubling the cap would project the KV cache past 85% of its size.
`more-kv-memory` is offered whenever requests queue at the cap with the KV cache that full, whether
or not queueing was diagnosed. A larger memory share gives the KV cache more room and may not
unblock the cap. When it is offered for that reason and the goal is not `tpot`, its command also
raises `max-num-seqs` to the cap the larger pool would hold, sized as `kv-sized-concurrency`
below, so that testing it can show a gain; its reason says so: "sized in KV tokens, the larger
pool would hold about 34 requests, so the command also sets max_concurrent_requests 34". Under `--goal tpot`
a raised cap worsens TPOT, so the command sets the memory share alone.

`kv-sized-concurrency` raises `max-num-seqs` to the cap the KV pool holds, not to the next power
of two. It is offered when requests queue at the cap and the sized cap is above both the current
cap and the one `raise-concurrency` would set. The cap is the number of requests that fill 85% of
the pool, at most the clients in flight. A request counts as the larger of the KV tokens a running
request held at the sampled peak and the workload's mean prompt plus answer tokens. The pool is
the one a cold start gives, because a new cap restarts the engine cold. It is taken, in this
order, from this run when it started cold; else from the newest cold start of the project's runs
with these settings (apart from the cap) and the same per-step token budget, since the budget
changes the pool; else from the smallest cold pool of these settings. A warm run with no cold
figure known uses its own. Its reason shows the sum: "sized in tokens (about
2,027 per request), 29 requests fill 85% of the 70,912-token pool a cold start gives (a new cap
restarts the engine cold), so 29 is worth trying". When it is offered, `raise-concurrency` leaves
Try first, so the section raises the cap once. It is not offered on a hybrid cache, where the
block count above sizes the cap. Under `--goal tpot` it is left out with `raise-concurrency`,
because more sequences per step worsen TPOT.

On a hybrid (linear-attention) cache, vLLM sizes every block to the state page. A raised cap
there is counted in whole blocks: `raise-concurrency` offers the demand up to the requests the
usable blocks hold (the pool less one reserved block, over the blocks one running request held),
whatever the KV-cache usage, and raising the cap is blocked only when no more requests fit. For
example 16 of 56 usable blocks free at 4 blocks per request raises a cap of 10 to 14. `more-kv-memory`
also raises `max-num-seqs` to the sequences the larger pool holds (its blocks over the blocks one
running request held in this run) when the cap held the running requests, widening a pinned CUDA
graph bound to match; at a share of 0.95 or more, it is listed under Not applicable here with the
pool's numbers. `trim-max-context-len` is listed there too (it adds no blocks), and so is
`fp8-kv-cache` when the longest request fits in one attention block (it doubles the tokens per
block, not the blocks); past that it is offered for the longer requests. `metrics.json` keeps
the tokens per attention block as `kv_block_tokens`.

A change whose reason reads no measured signal says where it comes from: "(from your settings)"
for `cuda-graphs`, "(from your prompts)" for `prefix-caching`, `ngram-speculation` and
`no-media-encoders`, "(from your prompts and settings)" for `enable-tool-calling`, and "(from
your answers)" for `compact-json`. They are
heuristics from the workload and the settings; `evidence:` grades the published support for the
change, not this run. `ngram-speculation` judges the median prompt length beyond what each
prompt shares with another prompt, over distinct prompts, so a prompt listed twice counts once.
It is also gated on this run's own answers: Tensward replays prompt lookup (the drafter n-gram
speculation uses: the latest earlier occurrence of the last three words proposes the next four)
over the run's prompts and answers, on the CPU, and counts the share of output words the drafts
would have supplied. Speculating adds a cost to every step that differs between models by
several times, so no break-even share is claimed: the offer moves to Could help and its reason
reads "Replaying prompt lookup on this run's answers, drafts would supply 41.0% of all output
tokens; measure it: the step overhead on this model is not known". Under `--goal tpot`, a p95,
the share is that of the answers it helps least (the lowest 5% of answers by share), because
those become the slowest once speculating. Under `--goal tpot` and `--goal throughput`, a pooled
share below 6% leaves it out ("below 6% speculation worsens TPOT"). Under `--goal ttft`, on a
prompt-bound run it is left out, because the draft tokens take step budget before waiting
prompts. Where the run cannot be replayed (no answers kept, or no step time) the reason
ends with "Prompt lookup could not be replayed on this run: ..." and the offer moves to Could help. When the workload declares `structured_output`, `ngram-speculation` is still offered, but under
Could help and last, always in full: grammar-constrained decoding often rejects the drafts, and its cost line
recommends `--require-equal`.

Shared prompt prefixes are judged on the prompts in the order the engine renders them: from the
engine's token ids in whole KV blocks when the run has them, else in the chat template's order
(a leading system message, then the tools, then the other messages). `stable-prompt-prefix` is
offered when at least a fifth of the prompts share a long text (256 words or more) that begins
within their first 5% of words but not at the first word, because a short span before it (for
example a date line in the system message) differs. It names the span and the shared length and
says to move the differing text after the shared text, so the prefix cache can reuse it. It
changes no server setting, so it has no command and no expected effect.

`compact-json` is offered under Could help when at least 5% of the answers that must be JSON ran
on whitespace until the token limit. It sets `--structured-outputs-config` with
`"disable_any_whitespace": true`, merged into any value the setup passes, and `"backend":
"xgrammar"` when the backend is unset or `auto` (vLLM 0.30 takes the option only with `xgrammar`
or `guidance`; another backend gets no offer). Answers can change in content, not only spacing:
a prompt the schema cannot hold may get a short, valid but meaningless JSON answer, and other
answers may get longer, so compare the answers before adopting it (`--require-equal` fails on
spacing alone). With a pinned backend a schema xgrammar cannot compile fails instead of falling
back. Its evidence is vLLM's documentation and Tensward's 0.3.6 GPU run, not calibration: on a
7B model it raised requests per second by 27% and took invalid JSON from 33% to 0%, mostly
because the prompt the schema could not hold ended early, while one healthy schema answer got
about twice as long.

`cuda-graphs` carries `--compilation-config '{"max_cudagraph_capture_size": N}'` when the
concurrency cap is known: N is the smallest power of two that holds the cap (the arrival's
concurrency or `max_inflight`, or `max_concurrent_requests` when lower), times `1 + k` when
speculating. Decode batches never exceed the cap, and capturing larger graphs costs memory and
start-up time. With no known cap, or a bound that reaches vLLM's default of 512, nothing is set.
Capture sizes you already pin are kept, and a later change that raises the concurrency or the
speculative tokens widens the bound.

Graph capture needs memory beyond the KV cache of one full-length request: the larger of 0.25 GiB
and 1.2% of the GPU's memory as vLLM reports it (measured: 0.22 to 0.26 GiB on a 22 GiB card, 0.08 GiB on a small
model). From the eager run's "Available KV cache memory", its KV capacity and `max_context_len`,
Tensward computes what is left. When that is less, `cuda-graphs` is not offered alone: it is
offered together with the first change that frees enough (`no-media-encoders`, a higher
`kv_memory_fraction`, or a smaller `max_context_len`; the media encoders count only when their
weights cover the shortfall), and when none applies the report lists `cuda-graphs` under "Not applicable here" with the numbers.

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

### Async scheduling off

With async scheduling (vLLM's default), the engine plans the next step while the GPU runs this
one, so a request that arrives meanwhile joins one step later than it would otherwise. When
prompts take long to prefill, that wait is a large part of TTFT. `no-async-scheduling` sets
`--no-async-scheduling`. It is offered only under `--goal ttft`, and only when:

- the setup does not already turn async scheduling off and does not speculate;
- the engine's own prefill time per request, the mean over the measured window from its prefill
  time histogram in the run's two metrics scrapes, is at least two decode steps (the run's mean
  step time).

The reason gives this run's figures and one measured cell, never a prediction for your GPU:
"prefill took 180 ms per request, about 4.5 decode steps of 40 ms. With async scheduling the next
step is already planned when a request arrives, so it waits one extra step; with it off, it joins
the next step. Measured once on a T4 with Qwen3-4B, not a prediction for this GPU: turning it off
cut prefill from 310 to 164 ms per request while each step grew from 117 to 122 ms". TTFT improves; requests per second and TPOT worsen, because
the GPU idles while the CPU plans each step, and the cost line says so (see
[Goals and declared limits](#goals-and-declared-limits)). Under `--goal ttft` the ladder is 512
tokens per step, async off, then 256 only when the longest prompt with the running decodes fits a
256-token step: with async off, a first token waits one step less, so `small-step-budget` is
offered from a TTFT p95 of 3 times TPOT p50 instead of 5 times, and async off comes before the
256 step when both are in Try first. On a power-capped T4 serving the `qa` workload (prompts up to
182 tokens, 128 running decodes) the ladder stops after async off, and the stop gives TTFT in
decode steps ("TTFT p95 431 ms is about 3.5 decode steps of 122 ms").

### Earlier runs, the plan and stopping

`analyse` reads the project's earlier complete runs of the same registered setup, so repeated
runs guide each other. The measure is the goal's: requests per second for `throughput`, TTFT p95
for `ttft`, TPOT p95 for `tpot`. The section opens with "So far:" when there is something to say:

- **Tried.** A change an earlier run measured is not suggested again. One line says what it did:
  "`raise-concurrency` (max-num-seqs=128) tried in run R: requests per second -4% against run P,
  kept in this run" (or "changed to max-num-seqs=64 in this run" when the flag has another value
  now, `reverted` when it is gone, or "it served no request"). The run P is the one the change was made from. At most the five most
  recent changes are listed, plus any that would otherwise have been offered; a group whose only
  change was tried reads "tried — see So far".
- **Rejected.** A change you rejected with [`tensward reject`](#reject-a-change-and-write-a-report-again)
  is never offered again for this project, matched by the engine arguments it changed, as a tried
  change is. It is listed once, as "`max-num-seqs=8` tried in run R: rejected by you (the reason
  you gave)". If the run being reported carries a rejected change, going back
  to the run it was made from comes first: "you rejected max-num-seqs=8 (run R): go back to run
  P's settings".
- **Revert first.** When the run's own last change made the goal's metric worse than the run it
  was made from by more than the noise threshold, the first suggestion under Try first is to go
  back to that run's settings, with the measured loss. The threshold is twice the largest spread
  between runs of identical arguments in the project, at least 2%; with no repeats it is 2% for
  throughput and 5% for TTFT and TPOT. Smaller moves are noise and are not called losses. While a
  setup is kept, a revert to the registered setup's run is `tensward set registered --project P`,
  since a plain `analyse` would run the kept setup.
- **Worse than the best run.** When an earlier run of other settings did better on the goal's
  metric than this run by more than the same threshold, a line says so: "this run is worse on TTFT
  p95 than run R, the best run so far: 431 ms there, 470 ms here (+9%)", and the keep section
  offers that run first (see [Keep a run's settings](#keep-a-runs-settings)).
- **Helped less than expected.** Under `throughput`, when the change's earlier report printed an
  expected effect and the measured requests per second came in under its low end, the line names
  the limit that took over ("`prefix-caching` helped less than run R expected (measured x1.10,
  expected at least x1.40): queueing before scheduling took over, the next target"). It is
  not printed when a revert is.

**The short plan.** The first change in Try first is followed by up to three lines labelled as
predictions: the limit likely to bind after it (under `throughput`, the estimate's binding limit,
else the entry's own "frees up" data), the signal that would show it, and the likely next change
for that limit. Each line is a prediction, and a later step never undoes an earlier one. Only the
first change gets a command. Under a latency goal the plan ends before a step whose limit does not
matter to the goal. The first line has one of three forms:

- the limit is not crossed in this run, so the next run can test it: `then (prediction): queueing
  before scheduling likely binds next (watch for requests waiting at the concurrency cap)`;
- the limit is crossed now, so no run can confirm it: `then: prefill compute likely still binds;
  KV-cache usage projected at about 83% (58,796 of 70,912 tokens) at a cap of 29`;
- no limit is left: `then: no other limit is crossed now; KV-cache usage projected at about 18%
  (64,771 of 352,320 tokens) at a cap of 32`.

A later step is named (`then (prediction): `X`, after which ...`) only while `X` is one of the
changes the report offers; the plan stops before the first that is not.

KV-cache capacity is predicted to bind only when the KV cache projected at the offered cap
reaches the classifier's warning level. Otherwise the plan names the crossed limit the change does
not address (under a latency goal, only a class that matters to the goal), and the projection
follows as its signal.

The next `analyse` that made that change, for the same goal, checks its first prediction against
the new run. There are three outcomes:

- "run R predicted queueing before scheduling would bind after `prefix-caching`: confirmed" (the
  form; no recorded run is quoted): the limit is crossed now and was not when the plan was made.
- "run R said prefill compute still binds after `kv-sized-concurrency`: it does (not a test: it
  was crossed when the plan was made)": the limit was crossed before the change, so the run
  shows nothing new. `report.json` has `held` true and `confirmed` false.
- "not confirmed: this run shows prefill compute (prompt processing took 82% of the engine's
  measured prompt capacity (4559 prompt tokens/s))", or "not confirmed: this run crossed no
  limit": the predicted limit is not crossed, which is the only outcome that counts against a
  prediction.

When the checked limit is decode memory bandwidth and the run speculates, the check says "can't
tell, as Tensward does not model the decode ceiling while speculating": `report.json` has
`cant_tell` true and `confirmed` and `held` false.

**Time per test.** A suggestion with a command says how long testing it takes here, from this
run's own timings: `testing it: about 6 min on this setup (this run: 85 s to load, 4.3 min to
warm up and measure)`. When the run held its concurrency cap and the change lowers it, the same
requests take longer at the lower cap, so the measured window is scaled by the ratio of the caps
and the line says so: `testing it: about 10 min on this setup (this run: 85 s to load, 4.3 min to
warm up and measure; at a cap of 8 instead of 16 the same requests take about x2.0 as long)`.
Loading and warm-up stay as measured.

**Stopping.** With no change left in Try first the report says why and whether more is possible,
and never just "nothing to change":

- the GPU was saturated: the finding in words (for example "busy 100%, at 68 of 70 W, SM clock
  1215 of 1590 MHz") and that server settings cannot add throughput; what can is less work per
  request, a smaller or quantized model or a faster GPU. When every request in flight was
  running, the clients set the batch instead: "the 128 clients set the batch (every request in
  flight was running), so the test, not a server setting, decides how much runs at once". Under
  `tpot` the report also names the trade-off left (`lower-concurrency`), and whether it was
  tried, is ruled out here, is applied in this run, or has no signal. Under `ttft` the stop
  does not cite the saturation, which limits throughput, not first-token time: "the first-token
  changes in this playbook were tried or do not apply here; a smaller step budget lets first
  tokens out sooner, but `small-step-budget` has no signal here; what else can help is ...";
- the knee is reached (a latency goal; see [Ladders](#ladders));
- every change Tensward has for this goal was tried or rejected: "Every change Tensward has for
  TPOT here was tried; the best run is R (TPOT p95 ...)" (with "Every Try-first change ..." when
  Could help still lists untried ones; "tried or rejected" when you rejected one of them);
- otherwise, which crossed classes have no change in this engine's playbook (or whose change is
  blocked or under Could help), which changes were left out because they worsen the goal or are ruled out
  here, and either how close prompt processing is to its limit (below) or that nothing measured
  says the GPU is at its limit, so a trace (`--trace`) or another goal may show where more is
  possible.

Under `--goal ttft` and `--goal tpot` every stop ends with "; no measured limit calls for a
change that leaves TTFT alone." (or "TPOT alone."): the promotion above has already been applied.
For example, the best-run stop reads "Every change Tensward has for TPOT here was tried; the best
run is 20261011T061548Z-36ac (TPOT p95 36.3 ms); no measured limit calls for a change that
leaves TPOT alone." The "For a saturated GPU" group is not shown above a latency-goal stop when it
would hold no change.

When prompt processing binds, the stop gives the headroom against the prompt rate the probe
measured, and the levers left: "prompt processing ran at 85% of the prompt rate this engine
measured (4,559 tokens/s), about 15% headroom for scheduling changes; what is left: prefix caching
(hit rate 1%), a smaller or quantized model, a faster GPU". Prefix caching is named when the hit
rate is under 10% or not measured. It needs the probe's rate; without it the stop falls back to
the trace advice above. When `raise-prefill-batch` has moved to Could help, the stop says so:
"for prefill compute and the per-step token budget, `raise-prefill-batch` is under Could help".
A crossed class whose change is blocked says so instead of "no change applies": "no change in this
engine's playbook applies to prefill compute; the change for queueing before scheduling is blocked
(see Try first); prompt processing ran at 85% of the prompt rate this engine measured (4,553
tokens/s), about 15% headroom for scheduling changes; ...".

### Ladders

A project that is tuned for a latency goal is a ladder: each run makes one change from the one
before. Four things keep a ladder honest.

**The trade since the starting settings.** Under `--goal ttft` and `--goal tpot` the first line of
So far is the cumulative trade, measured: "since run 20261010T224118Z-02ab (the latest run of your
starting settings): TPOT p95 46.4 ms -> 36.2 ms (-22%); requests per second 1.35 -> 0.429 (-68%);
TTFT p95 24,736 ms -> 131,055 ms (+430%)". A move whose printed percent is inside the noise
threshold of "Revert first" without repeats (2% for requests per second, 5% for a latency) reads
"(-1%, within noise)", one
that rounds to none "(unchanged)". It names the goal's metric, requests per second and the
other latency, each in the trade base and now, so a ladder that gives away most of its throughput
says so. The trade base is the newest run with the engine arguments of the project's first run, not
the first run itself: the first run starts cold and reads slower than a repeat. No number here is
an expectation. When the start-up wave (the first requests, sent at once and outside the
measured window) got more than 20% slower in TTFT p95, and both waves had two requests or more,
a second line says so: "the start-up wave (32 requests sent at once, not in the figures above)
waited longer: TTFT p95 4.1 s -> 6.3 s (+54%)" (with the request counts of both runs when they
differ). `report.json` keeps the trade as `goal.trade`:
`since` (the trade base), the metrics as `[base, now]` pairs and, for the start-up wave, `wave` (`before_ms`,
`after_ms`, `requests`, `base_requests`).

**The knee.** A step of a latency ladder is a knee when it gained less than a quarter of the
requests per second it cost: the goal metric improved by less than 0.25 times the share of requests
per second lost. The quarter is a rule of thumb, not calibrated against measured ladders. When the
newest such step is still in force, a change in Try first that moves one of its flags further the
same way leaves Try first for Could help, with the measured step at the front of its cost line:
"knee reached: the last step (max-num-seqs=8, run 20261010T224726Z-c96c to this one) moved TPOT p95
-10% for requests per second -44%; ...". If nothing else is left, the stop says so:

```text
Nothing left in Try first for TPOT: the knee is reached; the last step (max-num-seqs=8, run 20261010T224726Z-c96c to this one) moved TPOT p95 -10% for requests per second -44%; the change that would go further is under Could help; no measured limit calls for a change that leaves TPOT alone.
```

The next run's `Try:` command is still there to run, if the trade is worth it to you. The knee holds
until the flag changes again: a later run that changes only another flag does not clear it, and one
that sets the flag to a different value is judged from there.

**Predictions that can be tested.** The plan and its check (see above) say "still binds" when a
limit was crossed before the change, so a ladder does not count a prediction as confirmed or
missed when the run could not test it.

**Time per test.** A lowered cap stretches the measured window, so the estimate for the next test
is scaled by the ratio of the caps (see above).

A load-limited run gets none of these: its Try first is the advice to raise the load.

### Load-limited, step budget and prompt processing

Three classes say that the test, the per-step token budget or prompt work, not the server's
settings, held a run back. Each decides on the measures it has and names those it lacked
("not measured: ..."); "can't tell" is kept for a missing input that would decide.

- **The load generator** ("load-limited"). The server kept up: time-averaged running requests
  are at least 90% of the requests the test kept in flight, none waited (time-averaged waiting
  under 1), and no resource was near its limit (KV-cache usage under 80%, step tokens under
  80% of the budget, prompt work under half of the engine's busy time). The report says the
  test did not push the server hard enough and, under Try first, suggests raising its
  concurrency; no server change is prescribed. Any one of these rules the test out even when
  other measures are missing: requests waited, the server ran under 90% of the requests in
  flight, KV-cache usage reached 80%, steps carried 80% of the budget or more, prompt work took
  half of the engine's capacity. A saturated GPU also rules it out. The finding is "critical"
  only when every measure that was taken is well clear of its limit, and "can't tell" only when
  the running, in-flight or waiting averages are missing, or when neither the probe's
  prompt-work share nor the prefill ceiling is there. When the probe gave no capacity, the
  prefill rate against its datasheet ceiling stands in for the prompt-work share, at 25% or more
  (a lower bar than half, because that ceiling is a lower bound of the time prefill takes);
  the finding says so.
- **The per-step token budget.** Steps carried at least 90% of the engine's per-step token
  budget (`max_num_batched_tokens`, read from the engine's start-up log and recorded in
  `run.json`; unknown when the log does not say, and no budget is guessed) while requests
  waited. It is offered `raise-prefill-batch`; n-gram speculation is not offered under it. When
  the waiting gauge is missing but the steps carried under 90% of the budget, or no request
  waited, the class reads clear, and the finding names what was not measured.
- **Prompt processing, by share.** Prompt work (prompt tokens computed per second over the engine's
  prompt capacity, see below) is at least half of the capacity: the run is named prefill compute and
  ranked first.

When prompt processing or the step budget is the cause and the running requests stay under 90%
of the concurrency cap, queueing is a consequence, not a cause. It is then listed under the cause
as `also seen, as its symptom: queueing before scheduling: ...`. When the running requests sit
at the cap while others wait, the cap binds too: queueing stays a finding and `raise-concurrency`
stays in Try first.

**The prompt-capacity probe.** Prompt work is judged against the prompt tokens per second the
engine can compute, which Tensward measures instead of reading a datasheet. Before the workload,
outside its measurement window, `tensward analyse` sends your own prompts (system message, turns
and tools as the workload sends them), padded or cut in their last user turn to the median
prompt length (at most 4,096 tokens). Each starts with a unique line, so the prefix cache cannot
answer it, and asks for one output token. Enough are kept in flight to fill the step budget, for
about 24 s and up to 60 s. The rate is the prompt tokens of the requests that complete after the
first 3 s, over the time between the first and the last of them; at least 20 completions are
needed. The probe also warms the engine up. The value is cached in the project for the same
setup (model, GPU, step budget, concurrency cap, parallel degree, KV-cache dtype, prefix
caching, quantization, CUDA graphs, engine version and dtype), so the next run of that setup
does not probe again. `run.json` records the result under `prompt_capacity`, in `source`:

- `probed`: measured in this run.
- `cached`: reused from an earlier run of the same setup.
- anything else, such as `cannot-fill`, `too-few-completions`, `cached-prefix`, `short-prompts`
  or `failed`: the probe was refused or did not give a trustworthy rate. The capacity is then
  unknown, the run has no prompt-work share, and the reason is the `source`; the report shows
  it in words, for example "prompt capacity: not measured (a conversation does not end with a
  user turn), so there is no prompt-work share this run". A failed probe never stops the run.

The probe refuses a workload with records that end on a tool result (`not-user-turn`), so an
agent workload built that way runs without a measured prompt capacity; the stand-in above then
applies.

When tool calls are at least 20% of the answers and the requests that offered tools reach the
client, on average, at least 0.5 s after the engine's own mean time to first token (from its
time-to-first-token histogram), the Diagnosis section shows both means and says so: the tool-call
parser holds output until a call is complete, and API-server load adds to it. The two causes are
not told apart.

### Expected effect

Two changes can show what they are expected to do to this run's throughput (requests per
second, new over old): `prefix-caching` and `raise-prefill-batch`, on an NVIDIA A10G only. The
line follows the reason:

    expected: throughput at least x2.9 (this run: 98% of prompt tokens would be served from
    the cache; about 62 requests would run at once instead of 43, as many as the KV cache
    would have room for)

A range within a factor of 3 is printed whole ("+40% to x2.1"), a wider one by its low end ("at
least x2.9"); a range that starts at no change reads "unchanged to +30%", and one that includes
a loss prints the loss ("-4% to +30%"). The bracket says what the estimate read from this run.

The number comes from a model of the run's own limits. A change raises the number of requests
running at once up to the first of four limits: what the clients keep in flight, the concurrency
cap, the room in the KV cache and the per-step token budget. The bracket names the limit that
binds. The step time is split into prompt work, at the probed prompt capacity, and decode, and
requests per second never exceed the prompt capacity over the prompt tokens each request
computes. `raise-prefill-batch` adds that larger prompt chunks per step can raise the time per
output token. No line is shown below +5%, or when an input is unknown (for example the prompt
capacity or the step budget).

An estimate is made only where it was checked against measured before-and-after runs: an NVIDIA
A10G, a closed-loop workload on one GPU (no tensor or data parallelism), at most 2% of requests
failed, a steady measurement window of 30 s or more, no speculative decoding, no structured
output or forced tool call, and a model whose cache holds no linear-attention (Mamba) state.
`prefix-caching` also needs prompts without images, a model without routed experts and the engine's
token ids of every prompt (see `prompt_blocks.jsonl`); it replays the prefix cache over the run's
own send order and KV-cache size. `raise-concurrency` shows no number: on a held-out check its
printed low end was +7% and throughput fell 6%. Every other change, and every run outside these
conditions, shows no line.

Within each group of the report, the changes with an estimate come first, highest low end first, and
the others follow in their usual order. When an estimate's low end is at least x1.5, that change
moves to the front of Try first, under the bottleneck it relieves (KV-cache capacity when it frees
it, else prefill compute, for `prefix-caching`; the step budget for `raise-prefill-batch`). Critical
gates (such as fit and failed requests) stay above it. A change that may alter the answers, or one
offered only because its signal is near its threshold, is never moved by its estimate, and a change
that may alter the answers never comes before one that does not.

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

When the baseline is itself a run with `--engine-arg`, the section is titled "Run C compared
with run B" (B the baseline, C the run being compared) and names the baseline "run B" throughout,
not "your current setup".

The comparison, and the "What changed" block of `analyse`, lead with the goal's metric of the newer
run: requests per second, TTFT p95 or TPOT p95. A run from before 0.3.10 reads as the throughput
goal. When the two runs sent different workload samples (see [Workload JSONL](#workload-jsonl)),
`tensward compare` prints a note.

The output tokens per second in the speed table normally come from the requests. When the
window is steady, the engine's counters cover the same span and the engine's generated tokens
differ from the requests' by more than 10% (the same condition as the Checks line), the engine's
count over the window is used instead, and a line above the table names which of the two runs
use it. Runs from 0.3.5 on use it only on a window of 10 s or more. Runs recorded before 0.3.3 store no engine rate and use the requests' rate.

Each `analyse` run writes `run.json`: the snapshot id, the engine arguments and settings that
ran, the `engine`, its `engine_version` (`null` when it could not be read), the `platform`
(`nvidia`, or `null`) and the checkpoint `format`, whether responses were kept, the temperature, the runtime, GPU indices and image, and the
`VLLM_*` variables the engine inherited from the shell (only under `--runtime local`; docker
passes the `-e` variables, which are in the settings), and the `host` (see
[Run directory](#run-directory)). It also records each GPU in use
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
  answers cut short (`finish_reason` `length`), empty answers, answers that repeat themselves
  and invalid JSON (with `structured_output`) make the verdict "changed" when they get worse by
  more than 0.15 (reference) or 5 percentage points. An answer repeats itself when more than half
  of its 24-character pieces (case and spacing folded) already appeared earlier in it, as in a
  "be be be" loop. Structured answers (JSON records, numbered and HTML lists, tool calls written
  as text) were checked on synthetic examples only; they stay under half.

At temperature above 0 with a seed, repeats of a prompt sample the same answer, so the floor is
tight and any engine change that alters sampling shows as changed answers.

The two runs' compile caches are read from their start-up logs. When one started with a cold
cache and the other with a warm one, the comparison says so above the speed table: a cold start
can shrink the KV pool by up to a third, so KV-bound speed may differ for that reason alone.
Runs that recorded no compile-cache state are not compared on it.

When the baseline's report offered the change the candidate ran (the same `--engine-arg`
values) with an expected effect, the changes block after a changed `analyse` run adds a line
under the requests per second:

    - expected (before this change): throughput +20% to +40%; measured: +31%

The expected range is the one the baseline's report printed; the measured value is the change in
requests per second, in the same form ("+N%", or "xN.N" above x2). It is shown for comparison
only and changes no verdict.

### Equality gate

`--require-equal` (on `analyse` and `compare`) writes everything as usual, then exits with
status 4 unless the answers are equal. It is refused, before anything starts, with
`--no-retain-responses`, and on `analyse` without something to compare (`--engine-arg` or
`--baseline-answers`). A prompt is **identical** when every one of its successful candidate
answers is one of the baseline's answers for it: the same text, the same tool calls (arguments as
canonical JSON) and the same finish reason, over every answer, not a sample. When some prompts
call tools, the report adds "Tool calls identical for N of M prompts that call tools": a prompt
counts when the baseline or the candidate made a call on it, and its candidate calls match one of
the baseline's answers' calls (name and arguments, in order). The line is left out when nobody
called a tool, or when either side did not record calls (a run from before 0.3.3, or recorded
answers without `tool_calls`). The answers are
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

## Keep a run's settings

The setup registered with `init` (the `--current` command, or the configuration's serving fields)
is the **registered setup**. `tensward set RUN --project P` makes the engine arguments of run
`RUN` the **kept setup**: a plain `tensward analyse` and `tensward serve start --from current`
then start from them instead of the registered setup. `RUN` is a run id, or `last` for the
newest run in the project. Every analyse of the kept setup is still compared with the registered
setup's run. `set` writes `<project>/kept.json` (`0600`: `run`, its `engine_args`, the
`answer_changes` it makes against the registered setup, the `previous` kept run or `registered`,
the registration's `snapshot_id`, the run's `max_req_s_loss` as a share, and `set_at`) and leaves
the registration as it is, so earlier runs stay comparable. `inspect` shows the kept setup as `kept`.

```
$ tensward set last --project /home/me/p/lp --accept-answer-changes
kept the settings of run 20261011T061548Z-36ac (the latest run): tensward analyse starts from them
before: the registered setup (no engine arguments)
after:  --engine-arg quantization=fp8
this run changes the weight quantization, which can change answers; you accepted that with --accept-answer-changes
undo:   tensward set registered --project /home/me/p/lp
serve:  tensward serve start --project /home/me/p/lp --from current
a server already running keeps its settings until it is started again
```

- The fourth line appears only for a run that changes a setting that can change answers: the
  weight quantization, the KV-cache precision, speculative decoding or compact JSON. Such a run
  is kept only with `--accept-answer-changes` (`--accept-unreviewed` is accepted too). Without
  it, `set` exits 1:

  ```
  not kept: run 20261011T061548Z-36ac changes the weight quantization (quantization=fp8), which can change answers. Tensward does not judge answers: compare them in /home/me/p/lp/runs/20261011T061548Z-36ac/report.md ("Answers before / after"); if they are acceptable, keep it with:
    tensward set 20261011T061548Z-36ac --project /home/me/p/lp --accept-answer-changes
  ```

- When the kept run carries a change you rejected, a line before `undo:` says so: "you rejected
  this change (run R); `tensward reject --clear R --project P` takes that back".
- The last line appears only while a server of the project is running: it keeps its settings
  until `serve start` runs again.
- When the run's answers, or those of a run it was made from (one parent at a time), look broken
  against that run's own parent (see [Broken answers](#changes-that-can-affect-answers)), a second
  line warns: "warning: answers look broken in run R (at least 53 of 600 answers repeat
  themselves or are empty, against 0 of 600 in run B, ...): read them in .../report.md (Answers
  before / after)"; for an earlier run it adds ", which run S was made from". It does not stop the
  command. When `set` refuses such a run (no `--accept-answer-changes`), it prints the warning
  first, as "not kept: answers look broken in run R ...", then "Tensward does not recommend
  keeping run R. If you have read the answers and still want it:" and the command with
  `--accept-answer-changes`.
- The kept setup records the floor its run declared (`--max-req-s-loss`, see
  [A floor on requests per second](#goals-and-declared-limits)).
- `tensward set registered --project P` goes back to the registered setup, prints what it
  dropped ("dropped: --engine-arg quantization=fp8 (run R)") and the undo.
  Keeping a run with no engine arguments is the same, and says so. Keeping the run that is
  already kept writes nothing. `set registered`, or keeping a run with no engine arguments,
  writes `<project>/registered.json` (`0600`, the registration's `snapshot_id`); later steps are
  compared with the newest run of the registered setup; keeping another run removes it.
- A run is refused, exit 1 with one `not kept:` line, when it does not exist under
  `<project>/runs` (or, for `last`, the project has no run), was made from another registration,
  or served no request.
- A `kept.json` left from an earlier registration of the directory is ignored with a note;
  `tensward set registered` removes it.
- With a kept setup, `analyse --engine-arg ...` measures the given arguments as the whole change
  from the registered setup, as the suggested commands do; a note names a kept argument they
  leave out or replace.

Each report ends with how to keep its settings: the `tensward set` command (`set last` when the
run is the project's newest, its id otherwise), with `--accept-answer-changes` when needed, and
its undo. Under a latency goal a line before it repeats the measured trade since the starting
settings: "Measured since run R: gains: TTFT p95 -62%; costs: start-up wave TTFT p95 +23%; within
noise: requests per second -2%, TPOT p95 +3%." When an earlier run of other settings is the best
run so far on the goal's metric, the section names it with its `tensward set` command: first,
when this run is worse by more than the noise threshold ("This run is worse on TTFT p95 than run
R, the best run so far (431 ms against 470 ms in this run): `tensward set R --project P` keeps
that run's settings instead."), else after the command, as a run that "did as well ..., within
noise". The best run is never one below a declared floor. When the run's answers look broken,
a line before the command says so. The reject line reads `tensward reject last --project P
--reason "<why>"`. A run of the registered or the kept setup says so instead, under "Starting
setup", and the report's header and setup section name it "kept setup (run R)". A run that carries
a change Tensward rules out for the model or this GPU (an fp8 KV cache on Qwen2 or Qwen2.5, or
below compute capability 8.9) is not recommended for keeping: "Tensward does not recommend
keeping these settings: <the reason>. If you have read the answers and still want them:
`tensward set R --project P --accept-answer-changes`.", no change is offered on top of it (a
bottleneck with nothing else to offer says "held back until you go back, as the step above
says"), and going back to the run it
was made from (the registered setup when there is none) comes first in Try first.
A change that did not work is dropped with [`tensward reject`](#reject-a-change-and-write-a-report-again).

## Reject a change and write a report again

`tensward reject RUN --project P [--reason TEXT]` tells Tensward that a change did not work for
this project. `RUN` is a run id, or `last` for the newest run. The change is that run's engine
arguments beyond those of the run it was made from (the newest earlier run that sets only flags
this run sets, skipping runs that carry a change you rejected; the registered setup when there
is none). It is stored in `<project>/rejected.json` (`0600`: run, engine arguments, change, the
run it was made from, reason and time). After that, `analyse` and `report` never offer that
change again for this project, whatever the signal, and list it once under So far as
"`max-num-seqs=8` tried in run R: rejected by you (<reason>)". When every change Tensward has was
tried or rejected, the stop says so.

```
$ tensward reject last --project /home/me/p/lp --reason "answers got worse"
rejected the change of run 20261011T061548Z-36ac (the latest run): quantization=fp8 (made from run 20261011T054012Z-9d1e)
reason: answers got worse
tensward analyse and tensward report no longer offer it for this project
next:   tensward report 20261011T054012Z-9d1e --project /home/me/p/lp (the next change from run 20261011T054012Z-9d1e)
undo:   tensward reject --clear 20261011T061548Z-36ac --project /home/me/p/lp
```

- A reason that says the answer check did not run to a result (`judge: timeout`,
  `judge: judge_failed`, `judge: no output ...`, `judge: no status ...`, `judge: no_calibration`
  and the like) is no verdict on the answers: the first such rejection of a change does not drop
  it. The next report offers the same command once more with `--reuse-run R`, which copies run R
  (answers included) instead of measuring it again, while R measured exactly that step from the
  run being reported, kept its answers, is not itself reused, is at most 24 h old and was made by
  this minor version; otherwise with `--remeasure`. The report says so under So far: "...
  tried in run R: its answer check did not run to a result (judge: timeout), so it is offered once
  more". A second rejection of the same change drops it, whatever its reason.
- `tensward reject --list --project P` prints one line per rejected change, oldest first.
  `tensward reject --clear RUN --project P` takes the rejection of that run's change back.
  Give one of a run, `--clear` or `--list`; `--reason` goes with a run.
- `analyse --reuse-run RUN` also accepts a saved run explicitly. It must have measured the
  same engine arguments and environment, kept its answers, not itself reused another run,
  be at most 24 h old and come from the same Tensward minor version and registration.
  Otherwise `analyse` prints why it was not reused and measures the step. `--reuse-run` and
  `--remeasure` are mutually exclusive. Reusing the run copies its answers for another check;
  Tensward does not perform the answer check itself.
- Rejecting a run whose settings are the kept setup (or that measured it), or whose change is part
  of the kept setup ("run R made a change (max-num-seqs=8) that is part of the kept setup (run
  K)"), is refused, exit 1, with one `not rejected:` line that names the command to go back first,
  `tensward set registered --project P` (or `tensward set RUN` for another run). A run of the
  registered setup has no change to reject, and a run made from another registration is refused
  the same way.
- Every report's "To keep this change" block ends with the reject command for the run ("Not worth
  it? `tensward reject last --project P --reason ...` tells Tensward not to offer it again
  here."), or with the one that takes a rejection back.

`tensward report RUN --project P` writes a run's report again, without an engine or a new run, so
the next suggestion is available at once after a `reject` or a `set`. It rebuilds the run's
"What to try next" and "To keep this change" from the files in its run directory, with the runs
before it as its history and the kept setup and rejections as they are now, rewrites that run's
`report.md` and `report.json`, and prints what `analyse` printed. It makes no new run directory.
It refuses, exit 1 with one `not reported:` line, a run that does not exist, one made from
another registration, one written before 0.3.11 or whose analysis did not finish ("run R was
written before Tensward recorded what its report needs, or its analysis did not finish; run
`tensward analyse` again"), and one profiled with `--trace` or `--counters`, whose report depends
on the profile's analysis, which Tensward does not read back. A directory that is not a registered
project exits 2 with `project_not_registered`, as `set` and `reject` do.


### Start-up stop and stacks

A step whose KV pool would hold under half the tokens its parent used is stopped within seconds
of the server logging its pool, unless the step itself declares a smaller pool (a lower memory
fraction, context length or sequence cap, an explicit KV size, fp8 KV or fp8 weights). A server that exits for lack of memory for its KV cache is recorded the
same way. The stop is written to the step's `window.json` as `startup_guard` and shown in the
report, which offers the step again with an explicit KV size (`--engine-arg
kv-cache-memory-bytes=...`) worked out from the memory the log shows.

Two to four changes that do not interact are offered as one step: the first Try first offer
reads "Measured together" and carries one command with all of them. Offers are ordered by the
gain predicted for the goal against the risk of changing the answers. `report.json` carries it
as `answer_risk` (`low`, `medium` or `high`; any change of precision is `high` where the
workload offers tools). The stack's lossless changes are listed apart. A stack that gains less
than predicted, allowing for the noise band and the model's error, is reported as short and
its first change is offered alone.

`tensward reject` of a stack brings its changes back as single offers. A reason that starts
with `judge:` says that an answer check did not adopt the step: the stack is then rebuilt
without its changes that can alter the answers, and that remainder leads Try first. Any other
reason rejects that exact stack only; its changes are not offered together again.

### Knowledge records and additional options

Every analysed run writes `facts.json` (`facts_version`, a `digest` and the `facts`): the model,
the GPU, the engine and its settings, the workload and what the run measured, as data.
`tensward report` writes it for an older run that has none. `report.json` has a top-level `kb`
key: `{"pack": null}` when no knowledge source is installed, else the source's version, its
content digest, the digest of the facts, and the records it skipped with the reason (needing a
newer Tensward, naming a rule this Tensward lacks).

A knowledge source installed beside Tensward (see [extending](extending.md)) and the records
under `TENSWARD_KB_HOME` (default `~/.tensward/kb`, `local/records/*.json`) can explain an offer
("Why: KB record ... fired because ..." with the facts that fired it and what was measured), add
a "Worth measuring" section for changes that are documented but not yet measured here (each with
a command), withhold an offer a guard blocks (listed under "Not applicable here" with the
reason), and demote or remove an offer that a measurement showed to do nothing or to hurt. With
nothing installed and no local record the offers are the playbook's alone.

`tensward report RUN --project P --kb` prints, instead of the report, the knowledge source, the
facts digest, the records skipped, and for each offer the records that explain it (or "from the
playbook"), their evidence, and the records the run's facts could not decide.

A step measured from a parent appends one row to `<TENSWARD_KB_HOME>/local/rows.jsonl`: the
parent's facts, the change, what it did on the requests both runs sent, and the records that
offered it. It holds no prompt, answer or path, and Tensward uploads nothing.

Options added to `analyse`:

- `--engine-env NAME=VALUE` (repeatable) sets an engine environment variable for the run with
  `--runtime local`; the names are the ones the option lists in `--help`. It is refused with
  `--runtime docker` (a container gets only the registered setup's environment) and for a name
  the registered setup already sets. The run records it, printed step commands carry it,
  `tensward set` keeps it, and `tensward reject` accepts a run that changed only the environment.
- `--pin-client` keeps the engine off one CPU core so the load generator has a core of its own.
  With `TENSWARD_ENGINE_LAUNCHER`, the affinity is applied only to the launcher PID.
  An engine started separately does not necessarily inherit it.
- `--ttft-from first-content|first-token` (default `first-content`; applies only under
  `--goal ttft`): where the tool-call parser
  holds the first content back for 4 or more decode steps, offers rank on the engine's own time
  to first token and the report says so; `first-token` ranks on the first streamed chunk with a
  choice. The measured TTFT is the same either way.
- `--borrow-reference` reuses a reference run of the same setup up to 14 days old (24 hours
  otherwise) that ran under this machine's power cap, on this machine's own GPUs (not an
  imported run). A step from it is judged with the
  cross-launch floor widened (2% on requests per second and TPOT p95, 9% on TTFT p95).

Judged steps, those that can change the answers, measure at least 150 requests, even above the
configured count, as far as the prompt file reaches: the floor is the smaller of 150 and the
larger of the configured count and the file's prompts.

## Serve

`serve start --from current` (the default without an optimize result) runs the engine detached with the kept setup, else the registered setup (the registered
`--current` command, or the configuration's serving fields) so applications can call its
OpenAI-compatible API on `--host` (default `127.0.0.1`) and `--port` (default 8000). State and the
API key live under `<project>/serve/<name>/` (`--name`, default `default`; the served model name
is `tensward-<name>`). `serve start` waits until the model is loaded and answering (up to
`--ready-timeout`, default 900 s, printing progress lines on standard error), then exits; the
server keeps running until `serve stop`. The state is recorded as `starting` as soon as the
server is launched, so `serve status` and `serve stop` work while it loads. If `start` is
interrupted (Ctrl-C, SIGTERM), fails or times out, it stops the server it launched and marks the
state `failed`. A `state.json` that cannot be read is refused, by `start` too, with a message
naming the file; delete it and stop any server it described yourself. `serve status` exits 0 when the server is healthy and 1 otherwise.

`--engine-arg KEY=VALUE` (repeatable, as in `analyse`) serves the chosen setup with a change,
for example your current setup plus a recommended `max-num-seqs=64`. The state records the
overrides and `serve status` shows them.

`--from` is `current`, `latest` (the newest packaged result under `<project>/optimize/`, which
only the separate optimizer writes) or the id of one such result. Its `packages.json` format is
in [`extending.md`](extending.md#packagesjson). By default `start` serves
`latest` if the project has one, otherwise `current`, and prints which one it chose; when it
serves `latest` while a setup is kept, it says "a kept setup exists (run K): serve it with
--from current". A packaged result whose changes can affect answers is
served only with `--accept-answer-changes` (the earlier spelling `--accept-unreviewed` still
works).
