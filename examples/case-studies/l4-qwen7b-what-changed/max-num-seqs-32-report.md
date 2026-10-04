<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one edit: the box path `/home/ubuntu/p/<project>` (the project directory on the rented machine) was replaced by `<project>`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, usernames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 10, what changed: with `--engine-arg max-num-seqs=32`

How this was produced (2026-10-04):

- run id: `20261004T181807Z-63a5 (compared with baseline 20261004T181354Z-ec62)`
- GPU: NVIDIA L4 (24 GB), AWS g6.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.3.1 (released version, release validation run).
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0, `max-num-seqs` 8 as declared in the config)
- init command: `tensward init --project <project> --model <local copy of Qwen2.5-7B-Instruct-AWQ> --config examples/config.json --prompts examples/prompts.jsonl`
- analyse command: `tensward analyse --project <project> --engine-arg max-num-seqs=32`
- change: `--engine-arg max-num-seqs=32` (the first `Try:` command of the baseline report)

The text below the line is unedited apart from this header and the path edit named above. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261004T181807Z-63a5

Ran: vLLM 0.30.0 (docker image vllm/vllm-openai:v0.30.0) on NVIDIA L4 (driver 595.91.07, CUDA 13.2); checkpoint: safetensors, awq int4 group 128

## What changed vs your current setup (run 20261004T181354Z-ec62)

Change: --engine-arg max-num-seqs=32
- output 298.6 → 754.1 tok/s (+153%)
- requests 3.7 → 9.3/s (+153%)
- TTFT p50 5,502 → 174.9 ms (−97%)
- TTFT p95 5,800 → 692.4 ms (−88%)
- TPOT p95 22.9 → 37.0 ms (+61%, worse)
- answers: Answers unchanged within noise
- bottleneck: queueing before scheduling → none clear

One run each: repeat both runs before trusting a difference of a few percent.

## Diagnosis

Bottleneck: none clear (no signal crossed its threshold; some thresholds not yet calibrated on real GPUs)
- nearest: decode memory bandwidth: decode ran at 46% of the memory-bandwidth ceiling at the measured batch
- nearest: prefill stalling decode: TPOT p95 37.0 ms is 1.1x p50 33.2 ms
- not crossed: queueing before scheduling, decode memory bandwidth
- not crossed (uncalibrated thresholds): KV-cache capacity, prefill compute, prefill stalling decode, attention / long context
- speculation: off or not reported
- can't tell here:
  - GPU compute, tensor-bound kernels — kernel counters: run with `--counters`
  - host / CPU overhead — a GPU trace: run with `--trace`
- not modelled: offload / PCIe (needs offload signals, which come with the llama.cpp engine); multi-GPU communication (Tensward measures one GPU)
- quality: answers match the current setup's
- fit and failed requests: every request was served
- workload shape: 1.4 prompt tokens computed per generated token

## What to try next

Other changes the measurements support:
- `prefix-caching`: 2 of 10 prompts (20%) share a prompt prefix of up to 114 words with another prompt
  evidence: strong; may cost: a little GPU memory for the cache
  Try: `tensward analyse --project <project> --engine-arg max-num-seqs=32 --engine-arg enable-prefix-caching`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.

## Your current setup + overrides (max-num-seqs=32)

- source: declared in config
- settings: dtype=float16, max_concurrent_requests=32, max_context_len=8192, kv_memory_fraction=0.85, prefill_batch_tokens=2048, prefix_caching=False, tool_calling=True, tool_parser=hermes
- measured: output 754.1 tok/s, total 1855.9 tok/s, requests 9.28 req/s, goodput 9.28 req/s, TTFT p95 692 ms, TPOT p95 37.0 ms, 0 of 160 requests failed, hardware ceiling reached 59%

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 9.28 req/s
- output throughput: 754.1 tokens/s
- total throughput (prompt + output): 1855.9 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 9.28 req/s
- requests meeting the SLO: 100%
- TTFT p50: 174.9 ms
- TTFT p95: 692.4 ms
- TPOT p50: 33.2 ms
- TPOT p95: 37.0 ms
- end-to-end p95: 4885.7 ms
- start-up wave (first 32 requests, all starting together, outside the measured window): TTFT p50 644 ms, p95 1287 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 2.2%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 1
- preemptions: 0
- prefix-cache hits: 0 tokens
- prefix-cache hit rate: not measured
- KV cache capacity (measured by the engine): 227104 tokens, 27.7 full-length requests at once; estimated before the run: 214056 tokens (error -6%)

Tensward adds to every launch: --generation-config vllm (sampling follows the declared workload, not the checkpoint's generation_config.json), and the host, port, served model name and API key

## Engine defaults in effect (settings this setup leaves unset)

- async_scheduling: on, unless the setup is incompatible with it
- cuda_graphs: on

## Hardware ceilings (theoretical upper bounds, not targets)

- GPU: L4
- memory bandwidth: 300.0 GB/s (bandwidth: datasheet; device-derived cross-check 300.0 GB/s)
- dense tensor rate: 121 TFLOPS (datasheet)
- per decode step (one sequence): 4.48 GB of weights (the vision encoder and the embedding gather are not read)
- KV cache read per sequence at the average context: 7.6 MiB
- average context per sequence: 139 tokens
- average running batch: 27.9 sequences
- decode ceiling, one sequence: 67 tok/s
- decode ceiling at the measured batch: 1,777 tok/s
- decode measured: 823 tok/s
- prefill ceiling: 9,271 tok/s
- prefill measured: 1,162 tok/s
- decode, share of its ceiling: 46.3%
- prefill, share of its ceiling: 12.5%
- window time the tokens need at both ceilings: 58.9%
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Quantization

- checkpoint: awq int4 group 128
- kernel: MarlinLinearKernel (AutoAWQMarlinLinearMethod)

## Tool calling (quality, not performance)

- requests that offered tools: 32
- produced a tool call: 100%
- of the 32 that did, arguments parse as JSON: 100%
- of the 32 that did, tool is one of the offered tools: 100%
- of the 32 that did, arguments satisfy the tool's required keys and property types: 100%
- caveat: counts every request that offered tools; prompts that should not call a tool count as misses
## Compared with your current setup (run 20261004T181354Z-ec62)

**Answers unchanged within noise**

All answers identical to 20261004T181354Z-ec62 (text, tool calls and finish reason).
Your current setup reproduced its own answers for 10 of 10 prompts.

|  | current setup | this run | change |
|---|---|---|---|
| requests per second | 3.7 | 9.3 | +153% |
| output tokens per second | 298.6 | 754.1 | +153% |
| TTFT p50 (ms) | 5,502 | 174.9 | -97% |
| TTFT p95 (ms) | 5,800 | 692.4 | -88% |
| TPOT p95 (ms) | 22.9 | 37.0 | +61% |
| KV cache capacity (tokens) | 228,432 | 227,104 | -1% |

|  | current setup | this run |
|---|---|---|
| word-overlap similarity | 1.00 (its own repeats) | 1.00 (to your current setup) |
| answers with the same words | — | 100% |
| match with the reference | 0.72 | 0.72 |
| failed requests | 0% | 0% |
| answers cut short | 50% | 50% |
| empty answers | 0% | 0% |
| tool calls: produced a tool call | 100% | 100% |
| tool calls: valid JSON arguments | 100% | 100% |
| tool calls: known tool | 100% | 100% |
| tool calls: arguments match the schema | 100% | 100% |

A prompt counts as changed when its answers are more than 0.15 less similar to your current setup's than your current setup's are to each other (word overlap, at most 16 answers per prompt; a single changed number barely moves it, so read the answers below). Temperature: 0.

### Prompt `chat-01`

Your current setup:

```
A process is a running instance of a program, which includes the program code, data, and the execution context. Each process has its own memory space and resources, ensuring that processes are isolated from each other to prevent interference. Processes are managed by the operating system and can communicate with each other through inter-process communication mechanisms.

A thread, on the other han… (trimmed)
```

This run:

```
A process is a running instance of a program, which includes the program code, data, and the execution context. Each process has its own memory space and resources, ensuring that processes are isolated from each other to prevent interference. Processes are managed by the operating system and can communicate with each other through inter-process communication mechanisms.

A thread, on the other han… (trimmed)
```

### Prompt `chat-02`

Your current setup:

```
Certainly! To create a Python function that returns the n most common words in a text file, while ignoring case and punctuation, we can use the `collections.Counter` class for counting word frequencies and the `string` module for handling punctuation. Here's a step-by-step implementation:

1. Read the text file.
2. Normalize the text to lowercase.
3. Remove punctuation.
4. Count the frequency of e… (trimmed)
```

This run:

```
Certainly! To create a Python function that returns the n most common words in a text file, while ignoring case and punctuation, we can use the `collections.Counter` class for counting word frequencies and the `string` module for handling punctuation. Here's a step-by-step implementation:

1. Read the text file.
2. Normalize the text to lowercase.
3. Remove punctuation.
4. Count the frequency of e… (trimmed)
```

### Prompt `chat-03`

Your current setup:

```
To address missed sprint goals, consider these three concrete changes for the next two weeks:

1. **Improve Daily Stand-Ups**: Ensure daily stand-ups are more focused and productive. Start with a clear agenda, such as reviewing the sprint goals, discussing blockers, and assigning tasks. Use tools like the Eisenhower Matrix to prioritize tasks effectively.

2. **Enhance Task Breakdown**: Break down… (trimmed)
```

This run:

```
To address missed sprint goals, consider these three concrete changes for the next two weeks:

1. **Improve Daily Stand-Ups**: Ensure daily stand-ups are more focused and productive. Start with a clear agenda, such as reviewing the sprint goals, discussing blockers, and assigning tasks. Use tools like the Eisenhower Matrix to prioritize tasks effectively.

2. **Enhance Task Breakdown**: Break down… (trimmed)
```

All prompts, with every distinct answer: `compare/20261004T181354Z-ec62-vs-20261004T181807Z-63a5/answers.md`

Found something worth sharing, or a suggestion that was wrong? Tell us: https://github.com/Tensward/tensward/issues
