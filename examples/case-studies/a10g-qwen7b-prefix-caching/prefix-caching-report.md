<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 4, prefix caching: with `--engine-arg enable-prefix-caching`

How this was produced (2026-10-01):

- run id: `20261001T211247Z-60a9`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.6 (release validation run)
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0)
- `--current` command: `vllm serve /m/qwen --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes`
- analyse command: `tensward analyse --project ~/p --runtime docker --engine-arg enable-prefix-caching`

The baseline for this run is `../a10g-qwen7b-fp8-kv-cache/baseline-report.md` (run `20261001T210738Z-cfc9`). The command shown is reconstructed from the run's recorded engine arguments and the report's own `Try:` lines; the exact shell line is not stored with the run.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261001T211247Z-60a9

## Your current setup + overrides (enable-prefix-caching)

- source: imported from --current
- settings: max_context_len=8192, prefix_caching=True, tool_calling=True, tool_parser=hermes
- note: your --current command wins; these serving fields of the configuration are ignored: max_concurrent_requests (configuration 8; your command does not set it), kv_memory_fraction (configuration 0.85; your command does not set it), prefill_batch_tokens (configuration 2048; your command does not set it), prefix_caching (configuration False; your command does not set it)
- measured: output 1985.5 tok/s, total 4886.8 tok/s, requests 24.42 req/s, goodput 24.42 req/s, TTFT p95 326 ms, TPOT p95 14.1 ms, 0 of 160 requests failed, decode at 68.3% of its ceiling

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 24.42 req/s
- output throughput: 1985.5 tokens/s
- total throughput (prompt + output): 4886.8 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 24.42 req/s
- requests meeting the SLO: 100%
- TTFT p50: 60.4 ms
- TTFT p95: 326.4 ms
- TPOT p50: 13.4 ms
- TPOT p95: 14.1 ms
- end-to-end p95: 1987.9 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 1.3%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 0
- preemptions: 0
- prefix-cache hits: 16672 tokens
- prefix-cache hit rate: 87.7%
- KV cache capacity (measured by the engine): 248000 tokens, 30.3 full-length requests at once; estimated before the run: 243439 tokens (error -2%)

Tensward adds to every launch: --generation-config vllm (sampling follows the declared workload, not the checkpoint's generation_config.json), and the host, port, served model name and API key

## Engine defaults in effect (settings this setup leaves unset)

- prefill_batch_tokens: 2048 (8192 on GPUs with 70 GiB or more that are not A100, 16384 from 160 GiB), with chunked prefill on
- max_concurrent_requests: 256 (1024 on GPUs with 70 GiB or more that are not A100)
- kv_memory_fraction: 0.92
- async_scheduling: on, unless the setup is incompatible with it
- cuda_graphs: on

## Hardware ceilings (theoretical upper bounds, not targets)

- GPU: NVIDIA A10G
- memory bandwidth: 600.1 GB/s (bandwidth: derived from device (memory clock x bus width))
- dense tensor rate: unavailable (no published dense tensor rate for NVIDIA A10G)
- per decode step (one sequence): 4.48 GB of weights (the vision encoder and the embedding gather are not read)
- KV cache read per sequence at the average context: 8.7 MiB
- average context per sequence: 159 tokens
- average running batch: 22.7 sequences
- decode ceiling, one sequence: 134 tok/s
- decode ceiling at the measured batch: 2,908 tok/s
- decode measured: 1,985 tok/s
- prefill ceiling: not measured
- prefill measured: 357 tok/s
- decode, share of its ceiling: 68.3%
- prefill, share of its ceiling: not measured
- window time the tokens need at both ceilings: not measured
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
## Suggested experiments

No experiment is suggested by the measured signals.

## Compared with your current setup (run 20261001T210738Z-cfc9)

**Answers unchanged within noise**

All answers identical to 20261001T210738Z-cfc9 (text, tool calls and finish reason).
Your current setup reproduced its own answers for 10 of 10 prompts.

|  | current setup | this run | change |
|---|---|---|---|
| requests per second | 24.4 | 24.4 | -0% |
| output tokens per second | 1,987 | 1,985 | -0% |
| TTFT p50 (ms) | 62.2 | 60.4 | -3% |
| TTFT p95 (ms) | 325.2 | 326.4 | +0% |
| TPOT p95 (ms) | 14.0 | 14.1 | +0% |
| KV cache capacity (tokens) | 248,000 | 248,000 | +0% |

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

All prompts, with every distinct answer: `compare/20261001T210738Z-cfc9-vs-20261001T211247Z-60a9/answers.md`
