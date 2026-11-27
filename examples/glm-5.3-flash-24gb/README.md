# Run GLM-5.3-Flash on one 24 GB card

A 109 GB model on a single 24 GB card. On one L4 with 128 GB of RAM we measured about **12 tokens per second** on replies against 7.7 with the engine's defaults, and a 4,000-token prompt read in **15 seconds** against 139. Details and conditions are in [What we measured](#what-we-measured).

If you liked the recipe, we would appreicaitae a star!
https://github.com/Tensward/tensward

```
git clone https://github.com/Tensward/tensward
cd tensward/examples/glm-5.3-flash-24gb
./run.sh
```

That one command checks your machine, asks before downloading, starts the server and shows its speed. It stops with a clear message at anything only you can do.

## What you need

- Linux with one NVIDIA card of 24 GB, a recent driver, and the CUDA 12 runtime libraries (without them the prebuilt engine silently runs on the CPU; the kit checks).
- About 128 GB of RAM for the fast mode. About 64 GB works in a slower mode that reads part of the file from a fast local SSD. Sizes in between are untested.
- About 120 GB of free disk space.
- `bash`, `curl`, `tar`, `python3`. The kit never uses `sudo` and never installs packages.

This kit runs the public 2-bit model GLM-5.3-Flash (`UD-Q2_K_XL`, about 109 GB) with the public llama.cpp release b12722. It also works well behind a coding agent.

## After it is running

```
./run.sh chat      # talk to it in the terminal (--think also shows its thinking, dimmed)
./run.sh agent     # set up a coding agent (OpenCode)
./run.sh bench     # a few minutes of speed numbers, written to result.json
./run.sh status    # is it running?
./run.sh stop
./run.sh serve     # start it again (later starts are faster)
```

Every command ends with a short "Next:" list. `./run.sh help` shows all options.

The two modes: **pinned** (about 128 GB of RAM: the expert weights stay in RAM and the card holds a cache of the experts in use) and **mapped** (less RAM: the file is mapped and read from the SSD as needed). The kit picks by your RAM; `--mode pinned|mapped` overrides.

The first start in pinned mode measures the card once and starts twice (about 8 minutes). The size is saved, so later starts take about 3 to 4 minutes on a fast SSD. `--resize` measures again; `--cache-mib N` sets it yourself. The size is saved per preset, so the first start of another preset measures again: see [Presets](#presets).

`./run.sh check` also prints one `sudo` command for the pinned mode (`vm.compaction_proactiveness`): with a very large pinned area the kernel can keep a core busy and slow replies. Our measurements were taken with it set to 0, and on our machine replies were slower without it. You run it yourself; the kit continues without it.

## Presets

| Preset | What it changes | Answers change? |
|--------|-----------------|-----------------|
| `exact` | Experts held in RAM, expert cache on the card, tuned miss split. The model's own routing, no draft head. | No model change. The text can differ slightly from the engine's defaults |
| `fast` (default in pinned mode) | `exact` plus the engine's built-in draft head, one token ahead. Context and cache are sized so a long prompt cannot exhaust the card. | Slightly: the arithmetic differs, not the model |
| `fast --routing` | Our routing patch (needs the optional build). Replaces a missing expert by a cached one with a close score. The draft head is turned off: the two do not combine. | Slightly |
| `--experts N` | Uses N experts per token instead of the model's 8. | Yes, more. See below |
| `jobs4` | Four parallel jobs, a somewhat smaller cache. More tokens per hour overall, slower per job. | No model change |
| `mapped` | The low-RAM mode. One stream only: four jobs were slower than one. | No model change |

In the pinned mode, each choice is sized for the card on its own. The first start of a preset other than the default (and of each `--routing`, `--experts N` or `--ctx N` choice) measures the card again, so it takes longer than later starts of the same choice, and going back to a choice you have already started costs nothing extra. The `mapped` preset does no sizing.

Changing the preset, the cache size or the draft head changes the text slightly. Identical flags give identical text.

Keep the server's prompt cache on. Never pass `--cache-ram 0`: a coding agent then re-reads its long opening prompt on every turn.

### Fewer experts (changes answers)

```
./run.sh serve fast --experts 6
```

This uses 6 experts per token instead of the model's own 8. It is faster, but **answers change** and quality can drop. It is off unless you ask. Check the answers on your own tasks first.

## Optional: build with patches

The release alone gives you every preset except `--routing`, the warm start and the read-ahead. These three come from patches in `patches/`. They are our own patches for you to apply yourself. They are tied to llama.cpp b12722, they are not part of llama.cpp, and they are not submitted to it.

```
./run.sh fetch /path/to/folder --source
./run.sh build --patches routing,warm        # choose from advise, routing, warm
```

The build fetches the source at the pinned commit, applies only the patches you name (always in the order advise, routing, warm), builds for your card (the compute capability is detected), and installs beside the release. The built engine is then used automatically; `--engine release` forces the release.

Before you choose it: the build takes a while and uses all cores. It needs `cmake`, `make`, `patch` and the CUDA toolkit's compiler, `nvcc`. The compiler is taken from `$CUDA_HOME/bin`, `$CUDA_PATH/bin`, then `PATH`, then the usual install folders (`/usr/local/cuda/bin`, then `/usr/local/cuda-*/bin`, newest first, then `/opt/cuda/bin`), and the build prints which one it uses. `fetch --source` does not read model files again that an earlier `fetch` has already checked.

- `advise`: read-ahead for the mapped mode. In our test under a memory cap the first reply after a prompt was faster and steady reply speed was unchanged. It has not been measured on a real 64 GB machine yet. Do not use it in pinned mode.
- `routing`: switch for `fast --routing`. In one measured block on our own build: replies 4 to 8% faster for a 0.7% rise in perplexity. It changes answers.
- `warm`: saves the list of hot experts when the server stops and loads it at the next start, so the cache does not start empty. In our test the effect on the first tokens was within run-to-run variation.

## Warm start

```
./run.sh warm [--file some-text-of-yours.txt]
```

- Expert cache: `warm` generates a few hundred tokens so the cache fills with the experts in use. With `--file` it uses your text; without it, a short built-in coding question. It prints progress, not the reply. With the warm patch built, the list of hot experts is also saved at stop and loaded at the next start.
- Prompt cache: run `./run.sh agent --warm` from the folder you work in. The agent's opening prompt is long and fixed; it is read once and kept. It only helps if the next requests start with the same text, which is why the same folder and the same day matter: the opening prompt contains the folder and the date.

## Coding agent (OpenCode)

```
./run.sh agent          # writes opencode.json here, prints the start command
./run.sh agent --warm   # also sends one throwaway request to warm the prompt cache
```

The kit does not install any agent. If you do not have it: `npm install -g opencode-ai` (Node 22).

The agent sees the model under the short name `local/glm-5.3-flash`. An `opencode.json` of your own in the folder is never overwritten: the kit writes `opencode.json.new` beside it for you to merge.

## What the flags do

- `--jinja`: use the model's own chat template.
- `-fa on`: flash attention.
- `-np 1` / `-np 4 -kvu`: parallel jobs; `-kvu` shares one context memory.
- `-c 40960`: context length. A longer one with the draft head can run the card out of memory on a long prompt.
- `-ngl all --cpu-moe`: all layers on the card, expert weights in RAM.
- `--fit on` lets llama.cpp choose the placement itself; `--fit off` uses the placement given by the other flags.
- `-lm none`: load the model into RAM instead of mapping the file.
- `--moe-cache-mib`, `--moe-cache-cpu-share`: size of the expert cache on the card, and how misses are split between card and CPU.
- `-b 4096 -ub 4096`: big batches for fast prompt reading.
- `--chat-template-kwargs '{"reasoning_effort":"low"}'`: short thinking.
- `--spec-type draft-mtp --spec-draft-n-max 1`: the built-in draft head, one token ahead.

## Troubleshooting

- **"no CUDA card in use"**: the CUDA 12 runtime is missing and the engine would run on the CPU. Install it, or set `CUDA_LIB_DIR`, then run `./run.sh check`.
- **Out of card memory while serving**: use a smaller cache (`--cache-mib`) or context (`--ctx`), or use `exact`.
- **`cudaMalloc failed` right after a restart**: the old server's memory was not yet free. `./run.sh stop` waits for it; do not start before it reports stopped.
- **The build says `nvcc` is not installed**: the build needs the CUDA toolkit, not only the runtime libraries. If the toolkit is in an unusual folder, set `CUDA_HOME` to it.
- **The source download fails or restarts**: the source host does not support resuming. Run `fetch --source` again for one plain download. Model files resume.
- **Slow first prompt in mapped mode**: the file is read from the SSD the first time.
- **Replies slower than expected in pinned mode**: set `vm.compaction_proactiveness` to 0 (see `./run.sh check`).
- **A tool call is refused when testing with `ignore_eos`**: the engine does not allow `ignore_eos` together with `tools`.
- Logs: `.state/server.log`. A different release file name: set `ARCHIVE` (and `ARCHIVE_URL`) before `fetch`.

## What we measured

One NVIDIA L4 (24 GB), 128 GB of RAM, 16 cores, a local NVMe SSD; the public llama.cpp release b12722; one stream; `vm.compaction_proactiveness` set to 0. Figures were recomputed from the raw logs by an independent check. Your numbers depend on the card, the CPU, RAM speed and the SSD: run `./run.sh bench`.

| | Replies, tok/s | 256-token prompt | 4,000-token prompt |
|---|---|---|---|
| Release defaults | 7.7 | 15.4 s | 139 s |
| `fast` (the default preset) | 12.2 to 17.5 | 6.9 s | 15.0 s |
| `exact` | not yet checked | 6.7 to 6.9 s | 14.6 to 14.9 s |

- `fast` against the defaults: three default starts and two `fast` starts, alternating. The draft head accepted about 86% of its guesses. The exact text differs from the default engine's; we have not measured its quality separately.
- Coding agent, prompt cache on (a 13,000 to 15,000-token conversation): the turn after a helper request answered in 8.1 to 8.6 s. With the prompt cache turned off it took 59 to 63 s.
- `jobs4`: 1.43 times the reply tokens per hour of one job.
- `fast --routing` (our patch, our own build): replies 4 to 8% faster than the same build without it, for a 0.7% rise in perplexity, in one measured block.
- `mapped`, measured on another machine (one L4, 62 GB of RAM, 8 cores): 3.0 to 3.5 tok/s; a 4,000-token prompt in 81 s.
- The card was not throttled by heat. A 4,000-token prompt runs at the card's power limit.

Not measured yet: more than one client with the draft head, contexts longer than 40,960 tokens, other cards.

## Tested / not yet tested

Tested on a real machine: `check`, `serve` with the default preset on the public release, `bench`, `stop`.

Not yet tested from this kit: the downloads, `build` with patches, `agent`, `warm`, the mapped mode, `jobs4`, `chat`, `up` (plain `./run.sh`).

## Licences

- This kit: Apache-2.0 (this repository's licence).
- llama.cpp: MIT. The patches in `patches/` carry a pointer to its licence text (`patches/LICENSE-llama.cpp`).
- Model: MIT (the licence tag of the quantised model repository).
