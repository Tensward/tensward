# Set up Tensward (instructions for an AI coding agent)

Give this file to your AI agent: https://raw.githubusercontent.com/Tensward/tensward/main/AGENT_SETUP.md

---
You are an AI coding agent with a shell on the user's Linux GPU machine. Follow these steps in
order and run each check before moving on. Steps marked **ASK** need an answer from the user:
stop and ask, never guess.

## 0. What this is, and the ground rules

Tensward profiles LLM serving on the user's own GPU machine: it starts the user's engine (vLLM)
with their model, sends their prompts, compares with the GPU's theoretical ceilings and suggests
settings to try. It uploads nothing, has no telemetry and never tunes automatically.

Rules, for the whole session:

1. Never modify, stop or restart the user's production serving (containers, systemd units,
   processes). Tensward starts its own separate engine and needs a free GPU (step 1).
2. Never pass `--host` to `tensward serve start` unless the user explicitly asks.
3. Ask before any large download (model weights, the roughly 30 GB engine image) and before
   using `sudo`. Never use `--break-system-packages`. Do not install GPU drivers.
4. Do not pipe downloaded scripts into a shell (no `curl ... | sh`). If a tool is missing, ask
   the user to install it with their package manager.
5. If any command prints a refusal (a one-line JSON with a `code` on standard error) or fails,
   stop, show the user the exact message and ask. Do not work around it, and do not edit the
   user's model files, prompts or config to make a refusal go away without asking.
6. Report numbers honestly (step 9). Do not claim a speedup that was not measured.

## 1. Check prerequisites and the working directory

**ASK** first: "Where should I put models, inputs and the Tensward project? I suggest
`~/tensward-work`." Use the answer as `$TW` below (create it; models in `$TW/models`, inputs in
`$TW/inputs`, project in `$TW/project`). Then run these and keep the output:

```sh
uname -sm                      # Linux x86_64 (or aarch64)
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python3 --version              # 3.12 or newer is needed; uv can fetch one (step 2)
df -h $TW                      # free disk where the models and project will live
docker --version
docker info --format '{{json .Runtimes}} {{.DockerRootDir}}'
df -h "$(docker info -f '{{.DockerRootDir}}')"    # free disk where Docker stores images
```

OK looks like: `nvidia-smi` lists a GPU with room for the model plus KV cache (note which GPUs
are busy; if the only one serves production, ask which to use or when); about 30 GB free at
Docker's data root for the engine image, plus the model size in `$TW` (7B 4-bit about 5 GB, BF16
about 15 GB); `docker info` works without sudo and its runtimes include `"nvidia"`.

If a prerequisite fails, stop at the row below. Do not install drivers or Docker yourself and do
not use sudo without asking:

| Problem | What to do |
|---|---|
| No NVIDIA GPU or driver (`nvidia-smi` missing or fails) | Tell the user. Do not run `analyse` or `serve`. Only if the user wants to prepare, you may install and register (steps 2 to 6: `init`, `inspect` need no GPU); say that no measurement is possible here. Driver: https://www.nvidia.com/Download/index.aspx |
| Docker missing | Tell the user (https://docs.docker.com/engine/install/). If vLLM is installed on the host, offer `--runtime local` instead (confirm `vllm --version` works in the environment they serve from) |
| Docker permission denied on the socket | Tell the user to add themselves to the `docker` group (their action, then a new login), or offer `--runtime local` if vLLM is on the host |
| Docker lacks the `nvidia` runtime | Tell the user to install the NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html |
| Not enough disk | Tell the user how much is needed and how much is free, and ASK for another location with space (for `$TW`). Docker images need space at Docker's data root; that is the user's call to move |

## 2. Install Tensward

Prefer an isolated tool install, with whichever of these is already available:

```sh
pipx install tensward           # or: uv tool install tensward
tensward --help
```

`uv tool install` fetches a suitable Python itself if the system one is older than 3.12. If
neither exists, ask the user to install pipx or uv with their package manager, or use a
virtualenv:

```sh
python3.12 -m venv $TW/venv && . $TW/venv/bin/activate
pip install tensward
tensward --help
```

OK: `tensward --help` lists `init`, `inspect`, `analyse`, `serve` (`optimize` is a separate
commercial add-on: do not use it). If not found, open a new shell or run `pipx ensurepath`. If
`pip` says `externally managed environment`, use pipx, uv or a venv; never force it.

## 3. Gather inputs (ASK the user)

Ask these questions together, then wait.

1. **Model.** "Which model do you serve? Give me the local directory, or the Hugging Face name
   if it is not downloaded yet."
   - The directory must be a text-only safetensors checkpoint (BF16, FP16, AWQ, GPTQ,
     compressed-tensors or FP8) with `config.json`, `tokenizer.json`, `tokenizer_config.json`
     and `generation_config.json`. Models needing custom code (`trust_remote_code`) and
     multimodal models are refused.
   - To download: tell the user the name and approximate size, ask permission, then
     `hf download <org>/<model> --local-dir $TW/models/<model>`. The `hf` CLI comes from
     `pipx install huggingface_hub` (or `uvx hf ...`; inside a venv, `pip install
     huggingface_hub` in that venv). Gated models need the user to run `hf auth login`
     themselves; never ask for or handle their token.
   - `hf download` also leaves `README.md`, `LICENSE`, `.gitattributes`, a `.cache/` directory
     and tokenizer files. These are expected and `init` accepts them; do not delete them.
2. **Current serving command.** "What exact command do you run today to serve it
   (`vllm serve ...` or `docker run ...`)? Paste it, secrets removed." This becomes `--current`,
   the baseline everything is compared against; host, port, API key and served name in it are
   dropped. If they have no command, that is allowed (see step 4 for what the baseline is).
3. **Prompts.** "Do you have representative requests, for example a log of real traffic?" Real
   prompts give far better results than examples.
4. **Load.** "How many requests are typically in flight at once?" (`arrival.concurrency`); also
   the typical answer length (`output_tokens`), chat or plain completions, and tool calling.
5. **GPU.** Which GPU index, if there is more than one.

### Prompt file format

One JSON object per line (UTF-8), at most 10,000 records and 64 MiB. Each has a unique `id` and
either `prompt` (plain text) or `messages` (chat; must end with a user message). A chat record
may add `tools` (OpenAI function definitions), `tool_choice` and `max_tokens`.

```jsonl
{"id": "r1", "prompt": "a plain prompt"}
{"id": "r2", "messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "a chat prompt"}]}
```

All records must match the workload's `api` (step 4): `messages` for `chat`, `prompt` for
`completions`. Convert the user's logs with a small script into `$TW/inputs/prompts.jsonl`; keep only the
request text, never copy logs elsewhere, and ask before including anything sensitive. Use 50 to
200 varied prompts (requests cycle through them). `tensward init` (step 6) validates the file and
refuses bad input with `project_inputs_invalid` and a message naming the problem; rely on that
and show the user the message.

If the user has no prompts, download the example set. It is a data file (JSON lines, never
execute it). Say clearly in your final report that the results are illustrative:

```sh
mkdir -p $TW/inputs
curl -fsSL -o $TW/inputs/prompts.jsonl https://raw.githubusercontent.com/Tensward/tensward/main/examples/prompts.jsonl
```

It is a chat workload (ten records, one with tool calling): use `"api": "chat"`; the model needs
a chat template.

## 4. Write config.json

Create `$TW/inputs/config.json`. Read the checkpoint first:

```sh
python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print((c.get("quantization_config") or {}).get("quant_method"), c.get("dtype") or c.get("torch_dtype"), c.get("max_position_embeddings"))' $TW/models/<model>/config.json
```

This prints the quantization method, dtype and max positions. Fill `case.weight_precision` and `case.activation_dtype` from it (a wrong pair is refused with
`project_config_unsupported`):

| quantization | dtype | `weight_precision` | `activation_dtype` |
|---|---|---|---|
| none | bfloat16 | `bf16` | `bfloat16` |
| none | float16 | `fp16` | `float16` |
| `awq` (4-bit) or `gptq` | float16 | `int4` | `float16` |
| `fp8` or compressed-tensors FP8 | see below | `fp8` | as in the checkpoint |
| compressed-tensors W8A8 int8 | see below | `int8` | as in the checkpoint |

For 8-bit or compressed-tensors checkpoints use the `dtype` the checkpoint declares, try
`init`, and if it refuses with `project_config_unsupported`, show the user and ask. An
unquantized checkpoint whose dtype is neither bf16 nor fp16 is refused.

Workload fields (ask the user for anything you do not know):

- `workload.api`: `chat` or `completions`, matching the prompts (`chat` needs a chat template).
- `workload.output_tokens`: the most tokens generated per request.
- `workload.request_count`: requests to send, cycling through the prompts; 100 to 200 to start.
- `workload.arrival.concurrency`: the expected concurrent users, as
  `{"kind": "closed_loop", "concurrency": N}`.

**Which serving settings win (precedence).** The `case` serving fields (`max_model_len`,
`max_num_seqs`, `tool_calling`, ...) are optional.

- With `--current`, the command decides every serving setting, including tool calling
  (`--enable-auto-tool-choice --tool-call-parser ...`). The config's serving fields, including
  `tool_calling`, are ignored (`init` notes which ones). Leave them out.
- Without `--current`, the config's serving fields are the baseline. Add `tool_calling: true`
  only if the prompts offer `tools`.
- Without `--current` and without serving fields, the baseline is the engine's defaults,
  labelled "engine defaults (no current setup provided)".

Minimal valid example (chat, AWQ model, 32 concurrent users):

```json
{"schema_version": "1",
 "case": {"weight_precision": "int4", "activation_dtype": "float16"},
 "workload": {"api": "chat", "output_tokens": 128, "request_count": 160,
   "request_timeout_s": 300.0, "arrival": {"kind": "closed_loop", "concurrency": 32},
   "temperature": 0.0, "top_p": 1.0, "seed": 0},
 "warmup_requests": 4}
```

Unknown keys are refused. `engine_build`, `rounds`, `objective` and `workload.mode` are valid but unused by `analyse`; leave them out.

## 5. Pull the engine image (ASK first)

Skip with `--runtime local`. Tensward never pulls an image: use the image from the user's
`docker run` command, else `vllm/vllm-openai:v0.30.0`. If `docker image ls` lacks it, tell the
user it is about 30 GB at Docker's data root (give the free space you measured) and ask, then
`docker pull <image>`. Optionally confirm Docker sees the GPU: `docker run --rm --gpus all
--entrypoint nvidia-smi <image>`; if it fails, tell the user (step 1 table).

## 6. Register the project

```sh
tensward init --project $TW/project \
  --model $TW/models/<model> \
  --config $TW/inputs/config.json --prompts $TW/inputs/prompts.jsonl \
  --current "<the user's exact serving command>"
```

Omit `--current` if the user has none; for a long command use `--current-file PATH`. The
project directory is created private (0700) and must not be inside the model directory.
Repeating `init` with the same inputs is safe.

`init` only reads and hashes files (weights too: 1 to 2 minutes for 5 GB), starts nothing and
needs no GPU. OK: exit 0 and one JSON line on standard output with `"registration_state":
"registered"`, `current_setup` and `weights`. Read both to the user to confirm; notes there can
say the model path in their command differs from `--model`, or list config serving fields their
command overrides. Verify later with `tensward inspect --project $TW/project`.

A refusal is one JSON line on standard error with a `code` (exit 2: local environment, 3: input
contract). Stop and tell the user the code and message. Common ones:

| code | meaning, and what to do |
|---|---|
| `project_inputs_invalid` | config, prompts or `--current` missing or invalid. Show the message; fix files only with the user's agreement |
| `project_config_unsupported` | config asks for what the checkpoint lacks (precision pair, `max_model_len` above `max_position_embeddings`). Recheck step 4 |
| `project_inputs_changed` | an input differs from the registered one. Ask about a new `--project` directory |
| `project_layout_invalid` / `project_state_unsafe` | project directory is inside the model directory, collides with an input, or is not private. Choose another; do not chmod around it |
| `project_busy` | another registration is running; wait and retry |
| `checkpoint_*` | model files missing, unsupported, unsafe (symlink), changed during hashing, or wrong precision or quantization. Tell the user; never delete files in their model directory |
| `runner_failure` | I/O error such as permission denied; the message names the path |

Full list: https://github.com/Tensward/tensward/blob/main/docs/cli.md

## 7. Check the GPU is free

Tensward's engine reserves most GPU memory (vLLM default 92%). Confirm with the user that the
chosen GPU is free enough; if production occupies it, do not proceed: ask for another GPU or a
window after the user stops production themselves. GPUs chosen in the `--current` command are
the default; override with `--gpus INDICES`. The ceilings model one GPU.

## 8. Analyse

Only on a machine with a working NVIDIA GPU (step 1).

```sh
tensward analyse --project $TW/project --runtime docker
```

Use `--runtime local` if the user runs vLLM without Docker (`vllm` must be on `PATH`;
`--local-command CMD` overrides the server command). Add `--image <image>` only if it differs
from the one in their command. Optional: `--slo-ttft-ms N` and `--slo-tpot-ms N` (defaults 1000
and 100), `--no-retain-responses` to keep generated text out of the run directory.

This takes minutes (model load 1 to 5 minutes; 160 requests about a minute on an L4). Progress
goes to standard error. Do not kill it casually (Ctrl-C and SIGTERM stop the engine cleanly).

OK: exit 0, and standard output shows `current setup: ...` with output tok/s, total tok/s,
requests/s, goodput, TTFT p95, TPOT p95, `N of M requests failed`, and `hardware ceiling reached
NN%`, then `## Suggested experiments` and `run directory: <path>`. The full report is
`<run directory>/report.md`.

Failures print `tensward analyse failed: <cause>` and exit 1. Read the server log in the run
directory, show the cause to the user and ask. Typical: image missing (step 5), not enough free
GPU memory (step 7), model too large for the GPU. Exit 130 means interrupted.

## 9. Report back to the user

Read `report.md` and summarise it plainly:

- **Headline**: throughput (output and total tok/s, requests/s), goodput, TTFT p95, TPOT p95,
  failed requests, and the current setup source ("imported from --current", "declared in
  config" or "engine defaults (no current setup provided)").
- **Engine signals**: peak KV-cache usage, peak running and waiting requests, prefix-cache hit
  rate, and what they suggest (for example requests queueing while the KV cache is nearly empty).
- **Hardware ceilings**: theoretical upper bounds, not targets; a low share means headroom
  exists, not where. A prefill share can read low when prefix-cache hits are not counted.
- **Suggested experiments**: each with its reason and trade-off (usually TTFT against TPOT:
  "measure both"). They are experiments, not guarantees; the "declared load" is the concurrency
  in `config.json`, not measured traffic.
- **Checks** section, if present: say these make the run less trustworthy and why.
- If example prompts were used, say the results are illustrative only.

Then offer, do not assume, to try a suggestion. Each comes with a `Try:` command, for example:

```sh
tensward analyse --project $TW/project --engine-arg max-num-seqs=32
```

If the user agrees, run exactly that command (plus `--runtime local` if you used it), compare
the two reports (requests/s, TTFT p95, TPOT p95, failures) and report gains and costs. One run
is one data point: say so. Never change the user's production deployment; give them the flag.

Optional, only if asked: `--trace` adds a short profiled run (GPU busy and idle time;
diagnostic, never a speed claim). `--counters` is experimental (needs `--runtime local`, `ncu`,
root or `SYS_ADMIN`); read docs/cli.md#counters first and never use sudo for it without asking.

## 10. Serve (only if the user asks)

Needs a working GPU. Runs a server with the current setup, or plus a suggested change, for
applications to call. Never in place of production (use another GPU or port).

```sh
tensward serve start  --project $TW/project --engine-arg max-num-seqs=32
tensward serve status --project $TW/project
tensward serve stop   --project $TW/project
```

Leave out `--engine-arg` to serve the current setup unchanged. `start` waits until the model
answers (up to `--ready-timeout`, 900 s), then leaves the server running and prints the
endpoint (`http://127.0.0.1:8000/v1`), the api key file path and a `curl` command. Keep the
default loopback host; the API key file is mode 0600, so do not print or share it. `status`
exits 0 when healthy. Use `--port`, `--name`, `--gpus` if needed. When done, run `serve stop` and
confirm with `serve status` (`not running`).

## 11. Clean up and troubleshoot

- Tensward stops what it started, also on Ctrl-C. After a hard kill, remove leftovers only if
  labelled, never the user's own containers: `docker ps --filter label=tensward.run`, then
  `docker rm -f <container>`. A local detached server: `tensward serve stop --project $TW/project`.
- Everything lives in `$TW/project` (runs in `runs/<run_id>/`: `report.md`, `metrics.json`, logs)
  and `$TW/inputs`. Do not delete them without asking.
- Exit statuses: `0` success; `1` run or server failure; `2` local-environment refusal; `3`
  declared-input refusal; `130` interrupted.
- If you have to stop, tell the user the exact command that failed, its full message, the step
  you were on and what you tried. Reference:
  https://github.com/Tensward/tensward/blob/main/docs/cli.md
