<p align="center">
  <img src="docs/assets/tensward-icon.svg" alt="Tensward" width="96">
</p>

<h1 align="center">Tensward</h1>

<p align="center">
  Measures how your LLM is served on your own machine, offers one change at a time, and keeps a change only if it is faster and the answers pass your check.
</p>

<p align="center">
  <a href="https://pypi.org/project/tensward/"><img src="https://img.shields.io/pypi/v/tensward" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0"></a>
  <a href="https://colab.research.google.com/github/Tensward/tensward/blob/main/examples/notebooks/tensward-colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

You already run vLLM or llama.cpp. Tensward starts its own copy of your engine on a local port,
sends it a realistic workload (your prompts, or chat, agent, code and RAG presets, with tool
calls), reads the engine's metrics and the GPU, says what limits you, and offers one flag change.
It measures that change, compares the answers before and after, and keeps it in the project only
if it is faster. It never stops or reconfigures your running server, but it needs the GPU memory
free while it measures: if something holds it, it names the process and does not start. Nothing
is deployed until you export or serve it.

Free (Apache-2.0): one GPU, one `tensward try` at a time, every number measured on your machine.
Tensward Enterprise adds multi-GPU tuning, the automatic loop and A100 and H100 calibration:
[tensward.com](https://tensward.com).

## Two runs, one NVIDIA L4 each

Both are real terminal output from Tensward 0.4.2 on a GCP g2-standard VM with one NVIDIA L4
(22.5 GiB), one run each, 2026-11-07, shortened with "…". Numbers vary run to run. Tensward only calls
a change real when it is bigger than the differences between identical runs; details in the report.

### 1. llama.cpp: does not run, then runs, then faster

Qwen3.5-35B-A3B Q8_0 (GGUF, 34.4 GiB), `llama-server -ngl 99 -c 16384 -np 4`, llama.cpp build 11669
(CUDA), 4 concurrent requests, agent workload with tool calls.

| Step | Result |
|---|---|
| `llama-server … -ngl 99` | does not start: `Your model is 34.4 GiB and this GPU has 22.5 GiB, so your current setup won't start.` (`analyse`) |
| `llama-server …` without `-ngl` (llama.cpp's own `--fit`) | runs: 28.5 tok/s, 0.225 req/s, first word (TTFT) typically 2.82 s (slowest 1 in 20: 3.56 s); 41 layers on the GPU, experts of 18 layers on the CPU |
| `tensward try`: `--n-gpu-layers all --n-cpu-moe 17` | runs: 30.2 tok/s, 0.237 req/s, first word typically 2.65 s (slowest 1 in 20: 3.78 s) |
| `tensward try`: `--load-mode none` on top | 0.237 → 0.321 req/s (+33%); 44 of 64 answers identical, 20 reworded within the earlier run's own repeats: kept; host RAM 2.6 → 16.8 GiB |
| `tensward try`: `--threads 16` on top | 0.321 → 0.335 req/s (+5%); 24 of 64 answers reworded: kept after reading them (`try --keep`) |

Same L4 box, back to back, 2 runs each: llama.cpp's own `--fit` 0.225 req/s, 28.5 tok/s, first word 2.82 s; the kept setup 0.316 req/s, 41.4 tok/s, first word 1.50 s: **+40% requests/s, first word −47%**. Answers between the two: 98–101 of 160 identical, the same as two `--fit` runs (101 of 160). Host RAM: 6.0 → 20.5 GiB.

```text
$ tensward try
Trying --n-gpu-layers all --n-cpu-moe 17
Why: The model does not fit the GPU with every layer on it: the weights on the device take 33.9 GiB, the KV cache 0.4 GiB for 16384 tokens, and about 1.6 GiB go to the compute buffer and margin, against the GPU's 22.5 GiB.
…
Tried: --n-gpu-layers all --n-cpu-moe 17

                             now
────────────────────────────────────────────────────────────────────────────
Requests/s                   0.237
Output tok/s                 30.2
First word, typical          2.65 s
First word, slowest 1 in 20  3.78 s
Per token, typical           106 ms
Per token, slowest 1 in 20   153 ms
GPU active                   39% of the time
Decode vs ceiling            17%
VRAM used (of 22.5 GiB)      21.7 GiB
Host RAM used (of 62.8 GiB)  2.6 GiB
Experts on the CPU           17 layers
Answers                      160 of 160 non-empty, none repeating          ✔
Tool calls                   valid JSON 100%, right tool 100% of 82 calls  ✔

Your model did not run before; now it runs. Answers checked on their own.
Limited by reading the CPU-offloaded experts each step.
Kept in this project. Your running server is unchanged.

Next:
  tensward try          try --load-mode none
  tensward undo         drop this change
  tensward serve        stop tuning and serve what's kept now

$ tensward try
Trying --load-mode none (on top of your current setup: --n-gpu-layers all --n-cpu-moe 17)
Why: llama.cpp reads the CPU-offloaded experts through mmap; loading them into memory once is faster
This takes about 11 min.
…
Tried: --load-mode none (on top of your current setup: --n-gpu-layers all --n-cpu-moe 17)

                             before           after                change
────────────────────────────────────────────────────────────────────────────
Requests/s                   0.237            0.321                +33%
Output tok/s                 30.2             39.9                 +32%
First word, typical          2.65 s           1.46 s
First word, slowest 1 in 20  3.78 s           2.19 s               -42%
Per token, typical           106 ms           85.1 ms
Per token, slowest 1 in 20   153 ms           120 ms               -21%
GPU active                   39% of the time  52% of the time      +13 pts
Decode vs ceiling            17%              22%                  +5 pts
VRAM used (of 22.5 GiB)      21.7 GiB         21.7 GiB             same
Host RAM used (of 62.8 GiB)  2.6 GiB          16.8 GiB             +14.2 GiB
Experts on the CPU           17 layers        17 layers            same
Answers                                       44 of 64 identical   ✔
Tool calls                                    67 calls, all valid  ✔

Requests/s: +33%: a real gain.
20 answers reworded, no more than the earlier run's own repeats differ.
Limited by reading the CPU-offloaded experts each step.
Kept in this project (vs your command). Your running server is unchanged.

Next:
  tensward try          try --threads 16
  tensward undo         drop this change
  tensward serve        stop tuning and serve what's kept now

$ tensward try
Trying --threads 16 (on top of your current setup: --n-gpu-layers all --n-cpu-moe 17 --load-mode none)
Why: llama-server runs 8 threads of the 16 usable CPUs the host gives it; measured now: nothing else in Try first can run
This takes about 8 min.
…
Tried: --threads 16 (on top of your current setup: --n-gpu-layers all --n-cpu-moe 17 --load-mode none)

                             before           after                change
────────────────────────────────────────────────────────────────────────────
Requests/s                   0.321            0.335                +5%
Output tok/s                 39.9             42.3                 +6%
First word, typical          1.46 s           1.41 s
First word, slowest 1 in 20  2.19 s           1.7 s                no change
Per token, typical           85.1 ms          79.5 ms
Per token, slowest 1 in 20   120 ms           116 ms               no change
GPU active                   52% of the time  54% of the time      +2 pts
Decode vs ceiling            22%              not measured
VRAM used (of 22.5 GiB)      21.7 GiB         21.7 GiB             same
Host RAM used (of 62.8 GiB)  16.8 GiB         16.8 GiB             same
Experts on the CPU           17 layers        17 layers            same
Answers                                       40 of 64 identical   your call
Tool calls                                    67 calls, all valid  ✔

Requests/s: +5%: a real gain.
Limited by reading the CPU-offloaded experts each step.
Not kept: 24 of 64 answers are worded differently (none wrong). Nothing changes unless you keep it.
Compare them: ~/my-server/.tensward/runs/0003-threads-16/report.md (section "Answers before / after")

Next:
  tensward try --keep   keep it; nothing else to try after it
  tensward try          skip it and try --threads-batch 16
  tensward serve        stop tuning and serve what's kept now

$ tensward try --keep
Kept --threads 16 (answers reworded, you accepted).

No more changes to try.
```

`--load-mode none` is the gain: +33%, and the 20 reworded answers vary no more than the earlier
run's own repeats, so it is kept without asking. It holds 14 GiB more host RAM. `--threads 16`
adds 5% and rewords 24 of 64 answers, so it waits for you; kept here after reading them.

### 2. vLLM: does not start → starts → first word 19 s → 0.30 s

Qwen3.8-27B int4 (compressed-tensors, 18.1 GiB), `vllm serve /models`, vLLM 0.30.0, 8 concurrent
requests, chat workload, goal fast first token.

| Step | Result |
|---|---|
| `vllm serve /models` (defaults) | does not fit: a KV cache for 256 sequences needs 12.2 GiB, 1.9 GiB is left after the weights and overhead; `analyse` refuses to start it |
| `tensward try`: `--max-num-seqs 8 --max-model-len 4096 --gpu-memory-utilization 0.95` | runs: 70.7 tok/s, first word typically 19 s (slowest 1 in 20: 42.4 s) |
| `tensward try`: `--max-num-batched-tokens 512 --language-model-only --kv-cache-memory-bytes 2027414271` on top | first word typically 19 s → 0.30 s (slowest 1 in 20: 42.4 s → 0.31 s); 86 of 96 answers reworded |

```text
$ tensward try
Trying --max-num-batched-tokens 512 --language-model-only --kv-cache-memory-bytes 2027414271 (on top of your current setup: --max-num-seqs 8 --max-model-len 4096 --gpu-memory-utilization 0.95)
Why: the checkpoint takes images and video but no prompt sends any; serving only its text model stops the engine reserving memory for the media encoders and lifts the minimum batch size they impose (requests with images would then be rejected). Together with: TTFT p95 42385 ms is 602x TPOT p50 under a 2048-token step budget, and the longest prompt (95 tokens) with 5 running decodes fits a 512-token step: smaller steps let first tokens out sooner. …
This takes about 20 min.
…
Tried: --max-num-batched-tokens 512 --language-model-only --kv-cache-memory-bytes 2027414271 (on top of your current setup: --max-num-seqs 8 --max-model-len 4096 --gpu-memory-utilization 0.95)

                             before           after               change
───────────────────────────────────────────────────────────────────────────
Requests/s                   0.142            0.233               +64%
Output tok/s                 70.7             107                 +52%
First word, typical          19 s             299 ms
First word, slowest 1 in 20  42.4 s           311 ms              -99%
Per token, typical           70.4 ms          74.1 ms
Per token, slowest 1 in 20   71 ms            74.8 ms             +5%
GPU active                   98% of the time  98% of the time     same
VRAM used (of 22.5 GiB)      19.7 GiB         19.7 GiB            same
Answers                                       10 of 96 identical  your call

First word, slowest 1 in 20: -99%: a real gain.
Limited by a saturated GPU (before: limited by KV-cache capacity).
Not kept: 86 of 96 answers are worded differently (none wrong). Nothing changes unless you keep it.
Compare them: ~/my-server/.tensward/runs/0003-max-num-batched-tokens-512/report.md (section "Answers before / after")

Your model now runs at 70.7 tok/s. The last change is faster but rewords answers: keep it (tensward try --keep) or serve without it (tensward serve).
Next:
  tensward try --keep   keep it; nothing else to try after it
  tensward serve        stop tuning and serve what's kept now

$ tensward try --keep
Kept --kv-cache-memory-bytes 2027414271 --language-model-only --max-num-batched-tokens 512 (answers reworded, you accepted).

No more changes to try.
```

Trade-offs: `--language-model-only` turns off image and video input; 86 of 96 answers are
reworded, kept only after reading them (`try --keep`). The gain is measured against step 1, the
first setup that started; the defaults never ran.

### The answer check

Each try compares the answers with the setup before it.
- Identical, or differing no more than the earlier run's own repeats: kept if faster.
- Reworded beyond that (nothing broken): not kept; you read the report and `tensward try --keep` keeps it.
- Broken (empty, repeating, invalid tool call): refused, not offered for keeping.

Skipped a reworded change and want it back? `tensward runs` lists it; `tensward set <run id>
--accept-answer-changes` (for example `0003-threads-16`) makes that run's setup the kept one.

### Serve what is kept

```text
$ tensward serve
Serving your tuned setup (--n-gpu-layers all --n-cpu-moe 17 --load-mode none --threads 16 on top of your original command).
endpoint: http://127.0.0.1:8000/v1
api key file: ~/my-server/.tensward/serve/default/api_key

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $(cat ~/my-server/.tensward/serve/default/api_key)" \
  -H "Content-Type: application/json" \
  -d '{"model": "tensward-default", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}'

from openai import OpenAI
print(OpenAI(base_url='http://127.0.0.1:8000/v1', api_key=open('/home/you/my-server/.tensward/serve/default/api_key').read().strip()).chat.completions.create(model='tensward-default', messages=[{'role': 'user', 'content': 'Hello'}]).choices[0].message.content)

The server keeps running in the background after this command returns.
Stop it from any terminal: tensward serve stop
Run it as a service: tensward export --format compose (or systemd, helm)
```

Port 8000 taken (vLLM's own default)? `tensward serve --port 8001`.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have uv; then open a new shell
uv tool install tensward==0.4.2
```

Python 3.11 or newer; `uv` fetches one if your system has an older one. Other ways (pipx,
virtualenv, Ubuntu 22.04, offline) are in [docs/install.md](docs/install.md).

### Don't have an engine yet?

Tensward drives vLLM or llama.cpp; it does not install them. A `.gguf` file runs on llama.cpp, a
safetensors folder runs on vLLM. On Ubuntu 24.04 with an NVIDIA GPU
and its driver (`nvidia-smi` works), pick one.

Either way, add NVIDIA's CUDA apt repository once:

```bash
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
```

vLLM 0.30.0, the version Tensward 0.4.2 is validated on; it compiles kernels when it starts, so it
needs the CUDA 13.0 toolkit (about 3 GB), a C++ compiler, the ninja build tool and the Python headers:

```bash
sudo apt install -y cuda-toolkit-13-0 build-essential ninja-build python3-dev
uv tool install vllm==0.30.0
vllm --version
```

llama.cpp, the CUDA release build plus the CUDA libraries it loads:

```bash
sudo apt install -y cuda-cudart-12-8 libcublas-12-8 libgomp1
curl -fsSL -o llama.tgz https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-ubuntu-cuda-12.8-x64.tar.gz
mkdir -p ~/llama && tar -xzf llama.tgz -C ~/llama --strip-components=1
echo 'export PATH="$HOME/llama:$PATH"' >> ~/.bashrc && export PATH="$HOME/llama:$PATH"
llama-server --list-devices
```

If the download fails with a 5xx error, GitHub is having trouble; wait a minute and run it again.

The last line lists `CUDA0` with your GPU. Without a GPU, use
`llama-b11669-bin-ubuntu-x64.tar.gz` and only `sudo apt install -y libgomp1`. On a Mac:
`brew install llama.cpp`, then `llama-server --version` must show build 11669 or newer.

### Don't have a model yet?

Tensward never downloads models; it tunes the one you already serve. To get one, install the
Hugging Face CLI and download a model into a folder: `Qwen/Qwen3-8B` (safetensors) for vLLM, or
`Qwen/Qwen3-8B-GGUF` with a Q4_K_M file for llama.cpp. The commands, for Linux and Mac, are in
[docs/get-a-model.md](docs/get-a-model.md).

## Use it

Give init the command you'd serve with. Nothing needs to be running: Tensward starts its own
copy of the engine, and a running server would hold the GPU memory it needs.

```bash
tensward init --current "<the command you'd serve with>"   # then four questions and the first analysis
tensward try                       # measures the next offer; keeps it only if it is better and the answers pass
tensward try 2                     # offer number 2 instead
tensward undo                      # takes the last kept change back
tensward runs                      # every run: id, time, change, better or not, kept or reverted
tensward export --format compose   # a Docker Compose service for the kept setup (also systemd, helm)
tensward serve                     # serves the kept setup on a local port, prints a curl and a Python snippet
tensward serve stop                # stops it
```

For example, for the model from [docs/get-a-model.md](docs/get-a-model.md):

```bash
tensward init --current "vllm serve $HOME/models/Qwen3-8B"
tensward init --current "llama-server -m $HOME/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
```

`init` creates `./.tensward/` in the current folder and nothing else. Without `--current` it
takes your running server's command, or asks for one. It shows what it found (engine and
version, hardware, the model and whether it fits, the baseline command), asks for workload, tool
calls, peak concurrency and goal, and offers to run the first analysis, saying how many minutes it takes. Answers are numbered; Enter takes the default.

`try` takes one step per call and shows the time and GPU it needs first. It starts its own engine
on a local port and never touches your running server. "Kept" means kept in this project, not
deployed; `export` prints the command and flags as measured and deploys nothing.

No GPU? Run the flow on a free Colab GPU with a smaller model
([notebook](https://colab.research.google.com/github/Tensward/tensward/blob/main/examples/notebooks/tensward-colab.ipynb)).
Using a coding agent? Point it at [AGENT_SETUP.md](AGENT_SETUP.md).

## How is this different from `vllm bench` or guidellm?

A benchmark gives numbers. Tensward also compares them with the hardware's ceilings, says which
resource limits you and why, offers one change, and checks the answers and the tool calls before
keeping it. A faster setup that answers differently is not kept.

## What runs on your machine

- One engine process (`vllm` or `llama-server`) on 127.0.0.1 at a free port, started by Tensward
  and stopped when it exits, fails or you press Ctrl-C. It never stops your own server.
- It needs the GPU free (or free memory, on a CPU or Apple machine) and refuses to start when
  something else holds it.
- Disk: the project in `./.tensward/` (reports, run logs, retained answers) and the engine's
  compile caches in `~/.cache/tensward/`.
- Time: the first analysis takes a few minutes; `init` says how many for your model.
- Network: none of its own. No telemetry, no accounts. `init` and `analyse` download nothing and pull no
  image without asking; the engine you already have loads your model from disk.

## Uninstall

```bash
uv tool uninstall tensward && rm -rf ~/.cache/tensward ./.tensward
```

With the Docker runtime, the engine caches in `~/.cache/tensward/engine-cache` are written by
root, so remove them first with `sudo rm -rf ~/.cache/tensward/engine-cache`.

## Engines, hardware, models

| Engine | NVIDIA GPU | Apple Silicon | CPU only | Model formats |
|---|---|---|---|---|
| vLLM | yes | no | no | safetensors: BF16, FP16, AWQ, GPTQ, FP8, compressed-tensors |
| llama.cpp | yes (CUDA build) | yes (Metal, macOS) | yes | GGUF |

Run end to end (init, try, a kept change) in 0.4.2 on an NVIDIA L4: Qwen3.8-27B int4 and
LFM2-8B-A1B on vLLM; Qwen3.5-35B-A3B (Q8_0, Q4) and Qwen3.8-27B Q4 on llama.cpp. On Apple Silicon
(Metal, 0.4.1): Qwen3 GGUF, 0.6B to 30B-A3B. Other models are read and analysed, and the
report says when their family is not validated. The GPU list and details are in
[docs/cli.md](docs/cli.md#engines-and-hardware-reference).

## What it doesn't do

- It is not a serving engine. It drives yours and reads its metrics.
- One GPU, one step per `try`.
- It does not change your production setup or download models, and it pulls an engine image only after asking (or with `--yes`).
- A difference of a few percent between two single runs is not a finding: Tensward only calls a
  change real when it is bigger than the differences between identical runs.

## Privacy

Nothing is uploaded and there is no telemetry. Prompts and answers stay in your project folder,
which is private (`0700`). More in [docs/cli.md](docs/cli.md#privacy).

## Docs

- [docs/install.md](docs/install.md): install variants, offline install, uninstall.
- [docs/get-a-model.md](docs/get-a-model.md): download one model with `hf download` (safetensors for vLLM, GGUF for llama.cpp).
- [docs/cli.md](docs/cli.md): every command and option; `set`, `reject`, `compare`; explicit run
  ids; config and prompts formats; `--trace` and `--counters`; hardware ceilings; engine signals;
  Apple and llama.cpp details; MoE and image models; earlier case studies.
- [docs/errors.md](docs/errors.md): refusal codes and exit statuses.
- [ROADMAP.md](ROADMAP.md), [CONTRIBUTING.md](CONTRIBUTING.md), [docs/extending.md](docs/extending.md).

Apache License 2.0; see [LICENSE](LICENSE). Redistributions must keep the
[NOTICE](NOTICE) file (Copyright Adi Shik).
