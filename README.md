# Tensward

Tensward profiles and diagnoses LLM serving on your own GPU machine. It measures your current
setup under your own workload, compares the result with the hardware's theoretical ceilings,
shows where the GPU's time goes, and recommends engine settings to try. It is engine-agnostic by
design; vLLM is the first supported engine.

Using an AI coding agent? Point it at [AGENT_SETUP.md](AGENT_SETUP.md).

What it is not:

- It is not a serving engine. It starts your engine (vLLM) the way you already run it, offers
  your prompts to it, and reads its metrics.
- It uploads nothing and has no telemetry. Everything stays on the machine (see [Privacy](#privacy)).
- The open-source package does not tune automatically. It tells you what to try; you try it.
  (Tensward Optimize, a separate commercial add-on, adds automatic tuning.)

## Requirements

- Linux with an NVIDIA GPU and driver.
- Python 3.12 or newer (`uv tool install` fetches one for you if the machine has an older version).
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

Or into a virtualenv you manage (plain `pip install tensward` works there):

```sh
python3.12 -m venv ~/tw-venv && . ~/tw-venv/bin/activate
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
python3.12 -m venv .venv && . .venv/bin/activate
pip install .
```

## Quickstart

This uses Qwen2.5-7B-Instruct-AWQ, vLLM v0.30.0 and Docker, on a GPU with 24 GB. Apart from the
downloads, allow a few minutes: the model takes about 75 s to load and the workload below took
about a minute to run on an NVIDIA L4.

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

`tensward inspect --project ~/tw-project` re-checks the project against its inputs.

### 3. Analyse

```sh
tensward analyse --project ~/tw-project --runtime docker
```

Tensward starts the engine container, waits for the model to load (`--ready-timeout` defaults
to 900 s), runs the warmup, sends the workload, scrapes the engine's metrics, and always stops the
container, also on Ctrl-C. Loading takes minutes, so it prints progress lines on standard error
(launching, `still loading... 45 s` every 15 s, warmup, measuring at 25/50/75/100%, stopping).
When it finishes it prints the headline numbers, then the suggestions and the run directory
(with `--trace`, also a "GPU busy ... idle ..." line) on standard output:

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

followed by the suggestions, as in `report.md` (excerpt of a real run, see step 4):

```text
## Suggested experiments

- `raise-concurrency`: running requests hit max_concurrent_requests 16 with 16 waiting, and the highest sampled KV-cache usage is only 0.9%; at your declared load of 32 concurrent clients, highest sampled running plus waiting was 32, so 32 is worth trying (the load is what your configuration declares, not measured traffic). It usually cuts queueing (TTFT) but slows each token (TPOT) as more sequences share every step - measure both
  Try: `tensward analyse --project ~/projA --engine-arg max-num-seqs=32`
- `raise-prefill-batch`: TTFT p50 1674 ms is over 20x TPOT p50 20.1 ms with requests waiting (prefill-bound). Larger prefill chunks usually cut TTFT but can stall running decodes (TPOT p95) - measure both
  Try: `tensward analyse --project ~/projA --engine-arg max-num-batched-tokens=8192`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.
```

The run directory holds `report.md`, `metrics.json`, `requests.jsonl` (one row per request),
`responses.jsonl` (generated text; leave it out with `--no-retain-responses`), the raw metrics
scrapes and the server log (its last 10 MB; it keeps the startup lines, such as which kernels
were chosen). To judge against your own latency targets use `--slo-ttft-ms` and
`--slo-tpot-ms` (defaults 1000 and 100).

### 4. Read the report

Two full reports from real runs are in the repository: [`examples/report-l4.md`](examples/report-l4.md)
(the baseline) and [`examples/report-l4-suggested.md`](examples/report-l4-suggested.md) (the same
workload after following the first suggestion). Both are Qwen2.5-7B-Instruct-AWQ on an NVIDIA L4
with vLLM v0.30.0 in Docker, 160 chat, tool and RAG requests, and are unedited apart from a header
that says how they were produced. The parts that matter, from the baseline:

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
  2 s TTFT came from.
- **Hardware ceilings** are upper bounds from the GPU's memory bandwidth and tensor rate and the
  model's shape. No engine reaches them. A share far below 100% means there is headroom; it does
  not say where. Here decode reached 78% of its ceiling. Prefill shows 1.4% because prefix-cache
  hits are not counted as computed prefill, and this example cycles ten prompts, so most prompt
  tokens came from the cache (the report says so); with your own varied traffic it means more. On a GPU without a published
  dense tensor rate the prefill ceiling reads "not measured" and only decode is compared.
- **Suggested experiments** are settings worth trying, each with the evidence for it. Try one by
  running `analyse` again with the engine flag changed:

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
(`--name` changes it). Without `--from`, `start` serves the latest packaged optimize result if the
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
| GGUF | planned with the llama.cpp engine (see [ROADMAP](ROADMAP.md)) |

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
  ceilings count the experts each step actually reads (`k` of `E` per token, more at larger
  batches, assuming uniform routing), and prefill uses the active parameters.
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
contents; generated text is written only into your run directories (`--no-retain-responses`
turns that off). Project directories are created private (`0700`), and the API key for `serve`
is kept in a `0600` file and passed to the engine through its environment, never on a command
line.

## More

- [`docs/cli.md`](docs/cli.md): every command, option, file format and refusal code.
- [`ROADMAP.md`](ROADMAP.md): what comes next.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): how to build, test and extend Tensward.

## License

Apache License 2.0; see `LICENSE`. Redistributions must keep the `NOTICE` file
(Copyright Adi Shik).
