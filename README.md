# Tensward

Tensward profiles and diagnoses LLM serving on your own GPU machine. It measures your current
setup under your own workload, compares the result with the hardware's theoretical ceilings,
shows where the GPU's time goes, and recommends engine settings to try. It is engine-agnostic by
design; vLLM is the first supported engine.

Try it free: no GPU at hand? Run Tensward on a free Google Colab GPU in about 20 minutes: it measures an agent's vLLM setup, then follows two suggested changes in a row and shows what each one did.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tensward/tensward/blob/main/examples/notebooks/tensward-colab.ipynb)

Using an AI coding agent? Point it at [AGENT_SETUP.md](AGENT_SETUP.md).

Want to see real before/after reports first? See the [case studies](examples/case-studies/), including one where a change doubled the KV cache and garbled every answer.

What it is not:

- It is not a serving engine. It starts your engine (vLLM) the way you already run it, offers
  your prompts to it, and reads its metrics.
- It uploads nothing and has no telemetry. Everything stays on the machine (see [Privacy](#privacy)).
- The open-source package does not tune automatically. It tells you what to try; you try it.

## Requirements

- Linux with an NVIDIA GPU and driver.
- Python 3.11 or newer. On an older system (Ubuntu 22.04 has Python 3.10), use
  `uv tool install tensward`, which fetches a newer Python itself.
- For `--runtime docker` (the default): Docker with the NVIDIA Container Toolkit. The engine
  image must already be on the machine; Tensward never pulls images.
- For `--runtime local`: the engine installed on the machine (for vLLM, `vllm` on `PATH`).
- Model weights as a local directory. Tensward does not download models.

## Install

Install it as a command-line tool, in its own environment:

```sh
pipx install tensward           # or: uv tool install tensward
tensward --help
```

Tensward needs Python 3.11 or newer (`requires-python = ">=3.11"`). On an older system, such as
Ubuntu 22.04 (Python 3.10), pip finds no matching version; use `uv tool install tensward`, which
fetches a suitable Python itself.

Or into a virtualenv you manage (plain `pip install tensward` works there):

```sh
python3.11 -m venv ~/tw-venv && . ~/tw-venv/bin/activate
pip install tensward
```

On a current Debian or Ubuntu, `pip install tensward` outside a virtualenv fails with
`error: externally managed environment` (PEP 668: the system Python belongs to the package
manager). Do not force it with `--break-system-packages`; use pipx, uv or a virtualenv as above.
pipx comes from your package manager (`sudo apt install pipx && pipx ensurepath`). uv installs
with `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pipx install uv`); open a new shell
afterwards so `uv` and `uv tool` binaries are on your `PATH`.

From source:

```sh
git clone https://github.com/Tensward/tensward && cd tensward
python3.11 -m venv .venv && . .venv/bin/activate
pip install .
```

## Engines and hardware

Tensward detects the hardware (today an NVIDIA GPU, with its driver and CUDA version) and picks
the engine for the project when you register it, and says why:

- from the command in `--current`, when it is one an engine recognises (`vllm serve ...`, or a
  `docker run` of a vLLM image);
- from `--engine`, when you name one;
- otherwise automatically: the first engine that serves your checkpoint format on this machine.

`init` prints the choice and the reason. `analyse` and `serve` then use that engine, and refuse
before starting anything if its image or command is not present, saying how to get it.
`tensward env` shows what is detected and what works here. It starts nothing.

| Engine / hardware | Status |
|---|---|
| vLLM, safetensors checkpoints, NVIDIA GPUs | supported |
| llama.cpp, GGUF checkpoints | next |
| Apple Silicon: llama.cpp (Metal) and MLX | after that |
| SGLang | planned |

See the [ROADMAP](ROADMAP.md) for the order.

## Quickstart

This uses Qwen2.5-7B-Instruct-AWQ, vLLM v0.30.0 and Docker, on a GPU with 24 GB. Apart from the
downloads, allow a few minutes: the model takes about 75 s to load and the workload below took
about a minute to run on an NVIDIA L4.

### 0. Check the machine

```sh
tensward env
```

It lists the GPUs it detects, whether the vLLM image or the `vllm` command is present (and its
version), the checkpoint formats, and the combinations that work here. It starts nothing and
writes nothing. Run it first: if the image is missing, step 1 gets it.

### 1. Get a model, the image, and the two input files

```sh
hf download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir ~/models/Qwen2.5-7B-Instruct-AWQ
docker pull vllm/vllm-openai:v0.30.0
```

(`hf` is the Hugging Face CLI, shipped with `huggingface_hub`: `pipx install huggingface_hub`,
`uvx hf ...`, or `curl -LsSf https://hf.co/cli/install.sh | bash`. It replaced `huggingface-cli`.)
The directory must be a text or image+text safetensors checkpoint; see
[Supported models](#supported-models).

Tensward needs a serving configuration and a workload. Copy
[`examples/config.json`](examples/config.json) and [`examples/prompts.jsonl`](examples/prompts.jsonl),
then replace the prompts with your own traffic.

`config.json`:

```json
{
  "schema_version": "1",
  "case": {
    "weight_precision": "int4",
    "activation_dtype": "float16",
    "max_model_len": 8192,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.85,
    "prefix_cache": false,
    "tool_calling": true
  },
  "workload": {
    "api": "chat",
    "output_tokens": 128,
    "request_count": 160,
    "request_timeout_s": 300.0,
    "arrival": {"kind": "closed_loop", "concurrency": 32},
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0
  },
  "warmup_requests": 4
}
```

- `weight_precision` and `activation_dtype` must match the checkpoint (AWQ is `int4` with
  `float16`; an unquantized BF16 model is `bf16` with `bfloat16`, an FP16 one `fp16` with
  `float16`). `init` refuses a mismatch.
- `api` is `chat` or `completions`; every prompt record must use the same one.
- `request_count` requests are sent, cycling through your prompts; `output_tokens` is the most
  tokens generated per request; `arrival` says how they are sent: `closed_loop` keeps
  `concurrency` requests in flight, `open_loop` sends `rate_rps` per second, and `capped` sends
  `rate_rps` with at most `max_inflight` in flight (see [`docs/cli.md`](docs/cli.md)).
- Optional keys you can add: `engine_build` (a label for the engine release, recorded in the
  project identity), `rounds`, `objective` and `workload.mode`. `analyse` does not use them, so
  leave them out.
- Unknown keys are refused. The full format is in [`docs/cli.md`](docs/cli.md).

`prompts.jsonl` holds one JSON object per line: an `id`, and either `prompt` (plain text) or
`messages` (chat). A chat record may add `tools`, `tool_choice` and `max_tokens`; `labels` and
`reference` are notes for you. Two lines:

```jsonl
{"id": "chat-03", "messages": [{"role": "system", "content": "You are a helpful assistant. Answer clearly and concisely."}, {"role": "user", "content": "My team keeps missing sprint goals. What are three concrete things we can change in the next two weeks?"}]}
{"id": "tool-01", "messages": [{"role": "user", "content": "Do I need an umbrella in Rotterdam today?"}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "labels": ["tools"]}
```

The example file has ten records, including a tool-calling one and retrieval-style ones with a
long context. Your own traffic makes a far better test: use prompts of the lengths and mix you
actually serve. At most 10,000 records are accepted.

### 2. Register the project

`--current` is the command you run today. It is the baseline the report measures.

```sh
tensward init --project ~/tw-project \
  --model ~/models/Qwen2.5-7B-Instruct-AWQ \
  --config examples/config.json --prompts examples/prompts.jsonl \
  --current "vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --dtype float16 --max-model-len 8192 --max-num-seqs 8 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes"
```

`init` reads and hashes the inputs (the model weights too: about 1-2 min for a 5 GB model, so a pause is expected; `inspect` does the same) and starts nothing. It prints one line of JSON (identities and the
current setup, no input paths) that includes `"registration_state": "registered"`, what the
checkpoint is made of (`anatomy`) and whether it fits your GPU (`fit`). Things to know:

- The model path in `--current` can be anything (a Hub name, a container mount, a path on another
  host): `--model` is what Tensward measures, and `init` adds a note when the names differ. A
  different quantization than the checkpoint's is refused.
- Host, port, API key and served model name in `--current` are dropped because Tensward sets
  them. `docker run ... <image> ...` commands are accepted too.
- The project directory must not be inside the checkpoint directory.
- A refusal prints one line of JSON with a `code` on standard error; the codes are listed in
  [`docs/cli.md`](docs/cli.md).
- With `--current`, your command is the baseline and wins: the serving fields of `config.json`
  (`max_num_seqs`, `gpu_memory_utilization`, ...) are ignored, and `init` and the report list
  the ones that differ from your command. Without `--current`, the baseline is those serving
  fields. The `workload` section is always used.

`init` records the engine (see [Engines and hardware](#engines-and-hardware)) and prints the
choice and why. `analyse` and `serve` then use it, and refuse an engine image or command that is
not present, saying how to get it.

`tensward inspect --project ~/tw-project` re-checks the project against its inputs.

### 3. Analyse

```sh
tensward analyse --project ~/tw-project --runtime docker
```

Tensward starts the engine container, waits for the model to load (`--ready-timeout` defaults
to 900 s), runs the warmup, sends the workload, scrapes the engine's metrics, and always stops the
container, also on Ctrl-C. Loading takes minutes, so it prints progress lines on standard error
(launching, `still loading... 45 s` every 15 s, warmup, measuring at 25/50/75/100%, stopping).
When it finishes it prints the headline numbers, the bottleneck line, the next steps and the run
directory (with `--trace`, also a "GPU busy ... idle ..." line) on standard output:

```text
current setup: imported from --current
  output 724.7 tok/s
  total 1783.8 tok/s
  requests 8.91 req/s
  goodput 1.11 req/s
  TTFT p95 2111 ms
  TPOT p95 20.3 ms
  0 of 160 requests failed
  hardware ceiling reached 79%
```

`report.md` opens with `## Diagnosis`: the bottleneck that held the run back, its evidence and a
confidence, which classes did not cross their thresholds and what this run could not tell.
`## What to try next` follows: Try first, grouped by bottleneck, then Could help. The sections
below are from a real run: Qwen2.5-7B-Instruct AWQ on an NVIDIA L4 with the shipped example
workload, 32 clients against a concurrency cap of 8. The full report is
[`examples/report-l4-0.3.0.md`](examples/report-l4-0.3.0.md).
That report is from 0.3.0, before the thresholds were calibrated. Diagnosed again with the 0.3.1
thresholds, the same measurements read "confidence: high" with no calibration note, and the
decode-bandwidth line reads "(high)" too. Since 0.3.5 the "not crossed" list also names API-server
CPU, as below.

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

The run directory holds `report.md`, `report.json` (the same report as data), `metrics.json`,
`run.json` (what ran, on which GPU and host), `requests.jsonl` (one row per request),
`responses.jsonl` (generated text; leave it out with `--no-retain-responses`), the raw metrics
scrapes and the server log (its last 10 MB; it keeps the startup lines, such as which kernels
were chosen). To judge against your own latency targets use `--slo-ttft-ms` and
`--slo-tpot-ms` (defaults 1000 and 100).

### Did the change hurt the answers?

Run `analyse` on your current setup first (no `--engine-arg`), then on the change, for example
the first suggestion. The second report gains a section, "Compared with your current setup",
and the terminal prints its one-line verdict and where the answers are:

- the verdict, "Answers unchanged within noise" or "Answers changed: 2 of 10 prompts", with the
  speed change next to it;
- a speed table and a quality table: how similar the answers are, how many have the same words,
  the match with a prompt's `reference`, failed, cut-short, empty and (with `structured_output`)
  invalid-JSON answers, and the tool-call statistics;
- three prompts with the answer of each run, the ones that moved most first;
- a private `compare/<baseline>-vs-<candidate>/answers.md` with every prompt and its answers
  from both runs side by side, for you to read.

The baseline is the newest earlier run with no `--engine-arg` and the same registration. Answers
are judged against the baseline's own noise: a prompt counts as changed when the answers are
more than 0.15 less similar to the baseline's than the baseline's repeats of that prompt are to
each other (word overlap). That needs repeats, so set `request_count` in the configuration to at
least twice the number of prompts, run `tensward init` again, then analyse your current setup,
then the change; with every prompt run once the report shows the agreement and the answers but
makes no "unchanged" claim. Word overlap barely moves when one number in a long answer changes,
so read the pairs: Tensward does not judge whether an answer is correct.

`tensward compare --project ~/tw-project BASELINE_RUN CANDIDATE_RUN` compares any two runs of the
project by their ids under `runs/` and writes the same files. Both runs need their answers, so
runs made with `--no-retain-responses` or before this feature cannot be compared.

#### Answer-equality gate

When a change must not alter a single answer, add `--require-equal`: `analyse` and `compare`
still write everything, then exit with status 4 unless every answer to a prompt is one the
baseline gave for it. "Identical" means the text, the tool calls (as JSON, so key order does not
matter) and the finish reason, over every successful answer, and the report shows the first
differing character of each prompt. The gate also fails when a prompt got no answer, when more
requests failed, when there is no baseline, or when the baseline does not cover a prompt.

The gate relies on answers being reproducible across engine launches, which vLLM does not
promise: under load it is not batch-invariant, even at temperature 0. The report therefore says
how often your current setup reproduced its own answers, as evidence rather than a condition,
and when it did not, how to fix that. At temperature 0 (or with a seed), put
`VLLM_BATCH_INVARIANT=1` and `--no-enable-prefix-caching` in your `--current` command (the mode
is beta, needs compute capability 8.0 or higher, is slower, and does not support prefix caching
yet), run `tensward init` again, then analyse your current setup again. Without a seed at
temperature above 0, set one in the configuration first.

`--baseline-answers FILE` compares against recorded production answers instead of a run (see
[`docs/cli.md`](docs/cli.md#compare) for the format). Every `analyse` run also records each
GPU's name, driver, PCI device id, VBIOS and SM count; `--require-gpu NAME` and
`--require-driver VERSION` refuse a different machine before the model loads.

### 4. Read the report

Every report starts with one line saying what ran, for example `Ran: vLLM 0.30.0 (docker image vllm/vllm-openai:v0.30.0) on NVIDIA L4 (driver 595.91.07, CUDA 13.2); checkpoint: safetensors, awq int4 group 128`.

Full reports from real runs are in the repository, all Qwen2.5-7B-Instruct-AWQ on an NVIDIA
L4 with vLLM v0.30.0 in Docker:

- [`examples/report-l4-0.3.0.md`](examples/report-l4-0.3.0.md): 0.3.0, with the diagnosis, on the
  shipped example workload.
- [`examples/report-l4.md`](examples/report-l4.md) (the baseline) and
  [`examples/report-l4-suggested.md`](examples/report-l4-suggested.md) (the same workload after
  following the first suggestion): 0.2.0 output, before the diagnosis section, 160 chat, tool and
  RAG requests, unedited apart from a header that says how they were produced.
- [`examples/case-studies/l4-qwen7b-what-changed/`](examples/case-studies/l4-qwen7b-what-changed/):
  0.3.1, the same workload before and after `max-num-seqs=32`, with the "What changed" block.

More runs, including ones where a change did not help, are in
[`examples/case-studies/`](examples/case-studies/). The parts that matter, from the 0.2.0
baseline:

```text
## Your current setup

- settings: max_concurrent_requests=16, max_context_len=8192, kv_memory_fraction=0.9, tool_calling=True, tool_parser=hermes
- measured: output 724.7 tok/s, total 1783.8 tok/s, requests 8.91 req/s, goodput 1.11 req/s, TTFT p95 2111 ms, TPOT p95 20.3 ms, 0 of 160 requests failed, hardware ceiling reached 79%

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 0.9%
- highest sampled running requests (1 s polls): 16
- highest sampled waiting requests (1 s polls): 16

## Hardware ceilings (theoretical upper bounds, not targets)

- decode ceiling at the measured batch: 929 tok/s
- decode measured: 725 tok/s
- decode, share of its ceiling: 78.0%
- prefill ceiling: 9,272 tok/s
- prefill measured: 130 tok/s
- prefill, share of its ceiling: 1.4%
```

Following the first suggestion (`--engine-arg max-num-seqs=32`) and running `analyse` again gave,
on the same machine and workload:

| NVIDIA L4, Qwen2.5-7B AWQ, vLLM v0.30.0, 160 requests, 32 clients | baseline (`--max-num-seqs 16`) | `max-num-seqs=32` |
|---|---|---|
| requests/s | 8.91 | 14.75 |
| TTFT p95 | 2111 ms | 372 ms |
| TPOT p95 | 20.3 ms | 23.1 ms |

This is one example, not a promise: it is one run on one GPU with one workload, and more
sequences per step slowed each token a little (TPOT p95 rose) while cutting queueing. Your
numbers will differ; measure your own.

How to read it:

- **Your current setup** is the baseline: throughput, latency percentiles, and goodput (requests
  per second that met both the TTFT and TPOT limits). Here only 12% of requests did.
- **Engine signals** say why. Sixteen requests ran while 16 waited, and the KV cache was under 1% used:
  the engine had room, but `max_num_seqs` (16) held requests in the queue. That is where the
  2 s TTFT came from. Since 0.3.5 the section also shows the prompt tokens per engine step and
  "API-server CPU: N cores", the CPU the engine's API-server process used. Near one core the
  API server itself limits throughput, and the diagnosis names it (its thresholds are not
  calibrated yet).
- **Hardware ceilings** are upper bounds from the GPU's memory bandwidth and tensor rate and the
  model's shape. No engine reaches them. A share far below 100% means there is headroom; it does
  not say where. Here decode reached 78% of its ceiling. Prefill shows 1.4% because prefix-cache
  hits are not counted as computed prefill, and this example cycles ten prompts, so most prompt
  tokens came from the cache (the report says so); with your own varied traffic it means more. On a GPU without a published
  dense tensor rate the prefill ceiling reads "not measured" and only decode is compared.
- **Diagnosis** names the bottleneck, with its evidence and a confidence, and says what was ruled
  out and what this run could not tell. A run with fewer than 30 successful requests is capped at
  "possible". Queueing and decode memory bandwidth were calibrated on runs on an NVIDIA L4 and
  an A10G with vLLM 0.30, so a diagnosis of one of them can say "high". Every other class, host
  overhead and speculation included, tops out at "likely": its threshold was not yet measured on
  runs that avoid the bottleneck.
- **What to try next** lists engine settings, grouped by the bottleneck they address, each with
  its evidence grade and what it may cost. A change that does not apply to this model or setup is
  listed with the reason. Try one by running `analyse` again with the engine flag changed:

  ```sh
  tensward analyse --project ~/tw-project --engine-arg max-num-seqs=32
  ```

  The new report is headed "Your current setup + overrides (...)", and so is the terminal
  summary. Compare the two. Tensward suggests; it never changes your real deployment. The
  "declared load" in a suggestion is the concurrency your `config.json` declares; Tensward does
  not see your real traffic.

Two more things to know when reading it:

- **Unset settings run with the engine's default.** If your setup does not set a flag, the
  report lists what vLLM 0.30 does instead under "Engine defaults in effect". For example vLLM
  0.30 enables prefix caching by default, so a setup that never mentions it still shows prefix-cache
  hits and a hit rate under "Engine signals".
- **Tool-call rate counts every request that offered tools.** Prompts that should not call a tool
  count as misses, so a rate below 100% is not necessarily a model failure.

Other sections appear when relevant: the quantization kernel vLLM chose (`MarlinLinearKernel`
for AWQ is the fast path; a slow one is flagged), prompts that do not fit the context window, and
a "Checks" section when something makes the run less trustworthy.

### 5. See where the GPU time goes: `--trace`

```sh
tensward analyse --project ~/tw-project --trace
```

After the clean measurement, a separate short launch runs with the engine's profiler and the
report adds a GPU timeline. On the L4 run above:

```text
## GPU timeline (profiled - diagnostic only)

- profiled timing is not a speed claim: the profiler slows the engine, and this ran on a separate short launch after the clean measurement
- window: 2,218.9 ms, GPU busy 99.2%, 38,368 kernels
- 51 idle gaps of at least 50 us with no GPU activity: 0.6% of the window
```

Here the GPU was almost never idle, so the gap to the ceiling is inside the kernels, not in
scheduling. If the trace cannot be trusted, the section says "NOT TRUSTED" and no conclusion is
drawn from it. When the GPU is idle for 10% or more of the window, the report points to Tensward
Optimize for attributing the idle time to causes; this package reports the gaps, not the causes.
Profiled timing is diagnostic, never a speed claim; the profiler slows the engine.

### 6. Kernel counters: `--counters` (experimental)

```sh
tensward analyse --project ~/tw-project --runtime local --counters
```

`--counters` implies `--trace`, then profiles the three most expensive kernels with NVIDIA
Nsight Compute and reports a table per kernel: GPU time share, launch geometry, theoretical and
achieved occupancy, eligible warps, issue, DRAM and tensor-pipe utilization. Reading which limit a
kernel is at is left to you. It is experimental and has been verified only on an L4 and
an A10G. It needs:

- `--runtime local`, because `ncu` must wrap the engine process. With `--runtime docker` the
  report says "counters need ncu inside the image".
- `ncu` on `PATH` in the engine's environment (Nsight Compute 2025.3 or newer for CUDA 13).
- Permission to read GPU performance counters: root, or the driver option
  `NVreg_RestrictProfilingToAdminUsers=0`. If the machine forbids it, a qualification probe says
  so in the report and the rest of the analysis still runs. The engine and Tensward must be
  installed in the same Python environment.

Setup that worked (a Dockerfile with vLLM, ncu and Tensward, and the `docker run
--cap-add SYS_ADMIN` command): [docs/cli.md#counters](docs/cli.md#counters).

Excerpt of the table from a real L4 run (`--trace --counters`, 7B AWQ; the first two kernels of
three, verbatim; the full report lists all three, the qualification lines and the legend):

```text
| kernel | GPU time | launches | grid x block | resources | theoretical occ | duration us | achieved occ % | eligible warps | issue % | DRAM % | tensor % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `marlin::Marlin<1125899906910725l, 1125899906843648l, 1125899906910725l, 1125899906910725l, 256, 1, 8, 8, false, 4, 8, false>` | 62.7% | 753 | 58 x 256 (58 SMs) | 148 regs, 99 KiB smem | 16.7% (registers + shared memory) | 40 | 16.6 | 0.13 | 10.8 | 85.0 | 13.3 |
| `cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_16x16_128x1_tn_align8>` | 16.5% | 6 | 9,504 x 32 (58 SMs) | 128 regs, 8 KiB smem | 20.8% (shared memory) | 4,384 | 20.6 | 0.02 | 1.7 | 96.5 | 3.3 |
| `flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, false, false, false, false, true, false, true, false>` | 2.5% | 161 | 320 x 128 (58 SMs) | 252 regs, 80 KiB smem | 8.3% (shared memory) | 18 | 8.3 | 0.13 | 12.6 | 50.1 | 22.8 |
```

Here the int4 Marlin kernel that dominates decode shows 85% DRAM throughput and 13% tensor-pipe
use, which on this run is a memory-bandwidth-bound pattern. These are profiled numbers on one
GPU and one workload, not speed claims.

### 7. Serve

`serve` runs your current setup detached, for applications to call:

```sh
tensward serve start  --project ~/tw-project
tensward serve status --project ~/tw-project
tensward serve stop   --project ~/tw-project
```

`start` waits until the model is loaded and answering (it prints progress on standard error, and
`--ready-timeout` defaults to 900 s), then exits and leaves the server running. If you interrupt
it or it times out, it stops the server it launched. `status` and `stop` also work while a server
is still loading. To serve your current setup with a change a report suggested, pass the same
flag as to `analyse`:

```sh
tensward serve start --project ~/tw-project --from current --engine-arg max-num-seqs=32
```

`serve status` shows the overrides. `start` prints the endpoint (`http://127.0.0.1:8000/v1`; loopback unless `--host` is set), the
path of the API key file, and a ready-made `curl` command. The model is named `tensward-default`
(`--name` changes it). Without `--from`, `start` serves the latest optimize result if the
project has one, otherwise your current setup, and says which. `start` takes the same `--runtime`, `--image` and `--local-command`
options as `analyse`.

## Supported models

Tensward reads a checkpoint without loading it. Accepted: a text or image+text safetensors checkpoint, unquantized BF16 or FP16, or quantized with:

| format | status |
|---|---|
| BF16 (unquantized) | verified end to end on a real GPU (Qwen2.5-0.5B) |
| FP16 (unquantized) | registration checked; not yet served on a GPU |
| AWQ (4-bit) | verified end to end on real GPUs |
| GPTQ | registration checked against real published checkpoints; not yet served on a GPU |
| compressed-tensors (one config group: W4A16, W8A8 int8, FP8) | same as GPTQ |
| FP8 | same as GPTQ |
| GGUF | planned with the llama.cpp engine (see [ROADMAP](ROADMAP.md)); refused for now, with a message saying so |

Refused: custom code (`trust_remote_code`), files outside the standard
layout, symlinks. For latent-attention (MLA) and state-space hybrid models the hardware ceilings
are reported as unavailable. Tool-call parsers are chosen automatically for Qwen2/2.5, Mistral,
Llama 3 and Gemma 4 checkpoints. The exact layout and every refusal code are in
[`docs/cli.md`](docs/cli.md).

Validated end to end on real GPUs: Qwen2.5-7B-Instruct-AWQ on an NVIDIA L4 and an A10G with vLLM
v0.30.0. Gemma 4 26B-A4B (AWQ 4-bit) with vLLM v0.30.0:
- full `analyse` runs of the image workload example on an A10G;
- memory and KV capacity figures checked on an L4 and an A10G;
- the fit verdict checked on an L4, an A10G and a T4.
Other models and GPUs should work but have not been checked; please report what you find.

### Image workloads

A chat prompt may carry images next to its text, as OpenAI content parts. The image is a file
on disk, named by its path relative to the prompts file:

```json
{"id": "inv-017", "messages": [{"role": "user", "content": [
  {"type": "text", "text": "Extract the invoice total and due date."},
  {"type": "image_url", "image_url": {"url": "images/inv-017.png"}}]}]}
```

- Needs `"api": "chat"` and a checkpoint with a vision encoder. Only user messages carry images,
  at most 16 per prompt.
- **Paths only.** `https://` URLs, `data:` URLs, absolute paths and `..` are refused. If you
  copied an OpenAI example, save the image next to the prompts file and give its relative path.
- PNG, JPEG and WebP, checked from their headers. Animated PNG and WebP are refused. At most
  20 MiB and 40 megapixels per image, and 1 GiB of images per workload.
- `init` hashes every image into the workload identity. If one changes or moves afterwards,
  `inspect` and `analyse` name it, and a request never sends an image that differs from the
  registered one.
- The report counts image tokens as the engine did, and shows the requests with and without
  images separately.

[`examples/prompts-images.jsonl`](examples/prompts-images.jsonl) and its
[`images/`](examples/images) are a ready workload (invoice extraction, table reading, chart and
screen questions, one tool call); use it with
[`examples/config-images.json`](examples/config-images.json). `examples/images/make_images.py`
redraws the pictures. One record offers a tool: if your `--current` command does not enable tool
calling (`--enable-auto-tool-choice --tool-call-parser gemma4`), that request fails and
`analyse` suggests enabling it.

### Mixture of experts and image+text models

- **Mixture of experts** (Gemma 4 validated; Mixtral, Qwen3-MoE and others unvalidated): decode
  ceilings assume the fewest experts a step can read (`k` of `E`, whatever the batch), so no
  routing beats them; the report also shows how many a step reads with uniform routing. Prefill
  uses the active parameters.
- **Image+text checkpoints** register and can be analysed with text or image workloads (see
  [Image workloads](#image-workloads)). When no prompt sends an image, `analyse` suggests
  serving only the text model (`--engine-arg language-model-only`). The engine then reserves no
  memory for the image and video encoders and drops the minimum batch size they impose.
- **Fit**: `init` estimates whether the model fits your GPU before anything starts, and
  `analyse` reports the engine's measured KV capacity next to that estimate. With
  `--language-model-only` (or every media limit at 0) the estimate leaves the vision and audio
  weights out, as the engine does not load them.

## Supported GPUs

Hardware ceilings come from datasheet figures (dense, not sparse):

| GPU | memory bandwidth | FP16/BF16 tensor |
|---|---|---|
| L4 | 300 GB/s | 121 TFLOPS |
| A10 | 600 GB/s | 125 TFLOPS |
| T4 | 320 GB/s | 65 TFLOPS |
| A100 40GB | 1555 GB/s | 312 TFLOPS |
| A100 80GB (SXM) | 2039 GB/s | 312 TFLOPS |
| A100 80GB (PCIe) | 1935 GB/s | 312 TFLOPS |
| H100 SXM | 3350 GB/s | 989.5 TFLOPS |
| L40S | 864 GB/s | 362 TFLOPS |
| RTX 4090 | 1008 GB/s | 165 TFLOPS |
| RTX 3090 | 936 GB/s | 71 TFLOPS |

Other NVIDIA GPUs (for example the A10G): the decode ceiling is computed from the bandwidth
derived from the device's memory clock and bus width; the prefill ceiling is reported as
unavailable. Everything else in the analysis works on any NVIDIA GPU the engine supports.

## Privacy

Everything runs on your machine. Tensward makes no network requests of its own except to the
engine it started on localhost, and it has no telemetry and no accounts. It never pulls a Docker
image or downloads a model. A project stores where your inputs are and their hashes, not their
contents; generated text is written into your run directories and, when two runs are compared,
into `compare/` (both private), and the report quotes three answer pairs (`--no-retain-responses`
turns all of that off). Project directories are created private (`0700`), and the API key for `serve`
is kept in a `0600` file and passed to the engine through its environment, never on a command
line.

## More

- [`docs/cli.md`](docs/cli.md): every command, option, file format and refusal code.
- [`ROADMAP.md`](ROADMAP.md): what comes next.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): how to build, test and extend Tensward.
- [`docs/extending.md`](docs/extending.md): the extension API for packages that add commands
  or trace analysis.

## License

Apache License 2.0; see `LICENSE`. Redistributions must keep the `NOTICE` file
(Copyright Adi Shik).
