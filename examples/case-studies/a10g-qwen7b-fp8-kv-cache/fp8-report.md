<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 3, FP8 KV cache: with `--engine-arg kv-cache-dtype=fp8`

How this was produced (2026-10-01):

- run id: `20261001T211027Z-48df`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.6 (release validation run)
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0)
- `--current` command: `vllm serve /m/qwen --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes`
- analyse command: `tensward analyse --project ~/p --runtime docker --engine-arg kv-cache-dtype=fp8`

The command shown is reconstructed from the run's recorded engine arguments and the report's own `Try:` lines; the exact shell line is not stored with the run.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261001T211027Z-48df

## Your current setup + overrides (kv-cache-dtype=fp8)

- source: imported from --current
- settings: max_context_len=8192, kv_cache_dtype=fp8, tool_calling=True, tool_parser=hermes
- note: your --current command wins; these serving fields of the configuration are ignored: max_concurrent_requests (configuration 8; your command does not set it), kv_memory_fraction (configuration 0.85; your command does not set it), prefill_batch_tokens (configuration 2048; your command does not set it), prefix_caching (configuration False; your command does not set it)
- measured: output 1550.5 tok/s, total 3986.3 tok/s, requests 20.50 req/s, goodput 18.45 req/s, TTFT p95 325 ms, TPOT p95 15.8 ms, 0 of 160 requests failed, decode at 68.8% of its ceiling

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 20.50 req/s
- output throughput: 1550.5 tokens/s
- total throughput (prompt + output): 3986.3 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 18.45 req/s
- requests meeting the SLO: 90%
- TTFT p50: 60.3 ms
- TTFT p95: 324.7 ms
- TPOT p50: 14.0 ms
- TPOT p95: 15.8 ms
- end-to-end p95: 2047.4 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 0.6%
- highest sampled running requests (1 s polls): 31
- highest sampled waiting requests (1 s polls): 0
- preemptions: 0
- prefix-cache hits: 16672 tokens
- prefix-cache hit rate: 87.7%
- KV cache capacity (measured by the engine): 483632 tokens, 59.0 full-length requests at once; estimated before the run: 486878 tokens (error +1%)

Tensward adds to every launch: --generation-config vllm (sampling follows the declared workload, not the checkpoint's generation_config.json), and the host, port, served model name and API key

## Engine defaults in effect (settings this setup leaves unset)

- prefix_caching: on (for generative models)
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
- KV cache read per sequence at the average context: 4.3 MiB
- average context per sequence: 157 tokens
- average running batch: 17.1 sequences
- decode ceiling, one sequence: 134 tok/s
- decode ceiling at the measured batch: 2,255 tok/s
- decode measured: 1,550 tok/s
- prefill ceiling: not measured
- prefill measured: 299 tok/s
- decode, share of its ceiling: 68.8%
- prefill, share of its ceiling: not measured
- window time the tokens need at both ceilings: not measured
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Quantization

- checkpoint: awq int4 group 128
- kernel: MarlinLinearKernel (AutoAWQMarlinLinearMethod)

## Tool calling (quality, not performance)

- requests that offered tools: 32
- produced a tool call: 0%
- of the 0 that did, arguments parse as JSON: no tool call to judge
- of the 0 that did, tool is one of the offered tools: no tool call to judge
- of the 0 that did, arguments satisfy the tool's required keys and property types: no tool call to judge
- caveat: counts every request that offered tools; prompts that should not call a tool count as misses
## Suggested experiments

No experiment is suggested by the measured signals.

## Compared with your current setup (run 20261001T210738Z-cfc9)

**Answers changed: 10 of 10 prompts**

Not identical: 10 of 10 prompts differ (first difference at character 0 in tool-01).
Your current setup reproduced its own answers for 10 of 10 prompts.

|  | current setup | this run | change |
|---|---|---|---|
| requests per second | 24.4 | 20.5 | -16% |
| output tokens per second | 1,987 | 1,550 | -22% |
| TTFT p50 (ms) | 62.2 | 60.3 | -3% |
| TTFT p95 (ms) | 325.2 | 324.7 | -0% |
| TPOT p95 (ms) | 14.0 | 15.8 | +12% |
| KV cache capacity (tokens) | 248,000 | 483,632 | +95% |

|  | current setup | this run |
|---|---|---|
| word-overlap similarity | 1.00 (its own repeats) | 0.30 (to your current setup) |
| answers with the same words | — | 0% |
| match with the reference | 0.72 | 0.59 |
| failed requests | 0% | 0% |
| answers cut short | 50% | 44% |
| empty answers | 0% | 10% |
| tool calls: produced a tool call | 100% | 0% |
| tool calls: valid JSON arguments | 100% | — |
| tool calls: known tool | 100% | — |
| tool calls: arguments match the schema | 100% | — |

A prompt counts as changed when its answers are more than 0.15 less similar to your current setup's than your current setup's are to each other (word overlap, at most 16 answers per prompt; a single changed number barely moves it, so read the answers below). Temperature: 0.

### Prompt `tool-01`

Your current setup:

```
To determine if you need an umbrella, I first need to check the current weather conditions in Rotterdam. Let's get the weather information.

get_weather({"city": "Rotterdam", "unit": "celsius"})
```

This run:

```
(empty)
```

### Prompt `tool-02`

Your current setup:

```
lookup_order({"order_id": "A-20931"})
```

This run:

```
I'll check the status of your order with the ID A220233. Please

<|im_start|>-tools
<tools>
<tool>
<name>lookup_order</>
<parameters>
<object>
<property>
<name>order</
<value>A2023</>
</property>
</property
 </object>
 </parameters
 </tool>
 </tools>
```

### Prompt `chat-05`

Your current setup:

```
Certainly! Day one is a great start to your Lisbon adventure. We'll focus on food and viewpoints, keeping you on the flat areas as much as possible. Here’s a detailed itinerary:

### Day One: Exploring Lisbon's Best Food and Viewpoints

#### Morning: Belém and Pastéis de Nata
- **Start**: Head to Belém, a charming neighborhood known for its pastries and history.
- **Visit**: **Pastel de Belém**: T… (trimmed)
```

This run:

```
Certainly and
```

All prompts, with every distinct answer: `compare/20261001T210738Z-cfc9-vs-20261001T211027Z-48df/answers.md`
