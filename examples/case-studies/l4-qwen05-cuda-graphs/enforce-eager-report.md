<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and two edits: the box path `/home/ubuntu/p/<project>` (the project directory on the rented machine) was replaced by `<project>`, and the last line of the GPU timeline section, which pointed to a separate product, was deleted. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, usernames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 6, Qwen2.5-0.5B, CUDA graphs: with `--engine-arg enforce-eager`

How this was produced (2026-10-04):

- run id: `20261004T170351Z-d17c (compared with baseline 20261004T170036Z-6648)`
- GPU: NVIDIA L4 (24 GB), AWS g6.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.3.1 (release candidate, release validation run).
- model: Qwen/Qwen2.5-0.5B-Instruct (bf16)
- inputs: [`config.json`](config.json) (bf16, 96 requests, 32 concurrent clients, max-num-seqs 32) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (10 prompts, temperature 0, seed 0)
- init command: `tensward init --project <project> --model <local copy of Qwen/Qwen2.5-0.5B-Instruct> --config config.json --prompts examples/prompts.jsonl`
- analyse command: `tensward analyse --project <project> --trace --engine-arg enforce-eager`
- change: `--engine-arg enforce-eager`

Produced by the Tensward 0.3.1 release candidate. The released 0.3.1 words two Diagnosis lines differently: classes whose threshold is calibrated are listed under "not crossed:" apart from the uncalibrated ones, a run with no bottleneck says "some thresholds not yet calibrated", and the quality line follows the comparison's noise-aware verdict. Here the release candidate printed "quality: answers differ from the current setup's" while the comparison said "Answers unchanged within noise" (6 of 10 prompts differ byte for byte, but the current setup reproduced only 5 of 10 of its own answers); the released 0.3.1 prints "quality: answers differ from the current setup's only as much as its own repeated answers do". Numbers and the bottleneck diagnosis are identical. The GPU idle threshold is not calibrated in the released 0.3.1, so it prints host / CPU overhead with "confidence: likely", not "high".

The text below the line is unedited apart from this header and the edits named above. Numbers from one run on one machine are an example, not a promise.

Correction (Tensward 0.3.2): the throughput figures in this report were measured over a span that included the start-up wave's tail and the drain after the last request. In this case the steady window of the enforce-eager run is 1.57 s, too short to measure, so there is no corrected throughput figure and no corrected gain between the two runs: do not rely on the throughput figures or their ratio. TTFT and TPOT percentiles are unchanged. The report below is left as Tensward 0.3.1 wrote it.

---

# Tensward analysis 20261004T170351Z-d17c

Ran: vLLM 0.30.0 (docker image vllm/vllm-openai:v0.30.0) on NVIDIA L4 (driver 595.91.07, CUDA 13.2); checkpoint: safetensors, bf16

## What changed vs your current setup (run 20261004T170036Z-6648)

Change: --engine-arg enforce-eager
- output 2,828 → 1,098 tok/s (−61%, worse)
- requests 42.0 → 16.3/s (−61%, worse)
- TTFT p50 51.8 → 52.2 ms (+1%, worse)
- TTFT p95 120.3 → 199.2 ms (+66%, worse)
- TPOT p95 7.7 → 18.3 ms (+139%, worse)
- answers: Answers unchanged within noise
- bottleneck: decode memory bandwidth → host / CPU overhead

One run each: repeat both runs before trusting a difference of a few percent.

## Diagnosis

Bottleneck: host / CPU overhead (confidence: high)
- evidence: the GPU sat idle 71% of the traced window
- not crossed (uncalibrated thresholds): queueing before scheduling, KV-cache capacity, prefill compute, prefill stalling decode, decode memory bandwidth, attention / long context
- speculation: off or not reported
- can't tell here:
  - GPU compute, tensor-bound kernels — kernel counters: run with `--counters`
- not modelled: offload / PCIe (needs offload signals, which come with the llama.cpp engine); multi-GPU communication (Tensward measures one GPU)
- quality: answers differ from the current setup's (see the comparison below)
- fit and failed requests: every request was served
- workload shape: 1.3 prompt tokens computed per generated token

## What to try next

For host / CPU overhead:
- `cuda-graphs`: CUDA graphs are disabled; enabling them usually speeds decoding at the cost of longer startup and some GPU memory
  evidence: strong; may cost: a longer start-up and some GPU memory
  Try: `tensward analyse --project qwen05`
Other changes the measurements support:
- `prefix-caching`: 2 of 10 prompts (20%) share a prompt prefix of up to 114 words with another prompt
  evidence: strong; may cost: a little GPU memory for the cache
  Try: `tensward analyse --project qwen05 --engine-arg enable-prefix-caching --engine-arg enforce-eager`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.

## Your current setup + overrides (enforce-eager)

- source: declared in config
- settings: dtype=bfloat16, max_concurrent_requests=32, max_context_len=8192, kv_memory_fraction=0.85, prefill_batch_tokens=2048, prefix_caching=False, cuda_graphs=False, tool_calling=True, tool_parser=hermes
- measured: output 1097.9 tok/s, total 2978.6 tok/s, requests 16.31 req/s, goodput 16.31 req/s, TTFT p95 199 ms, TPOT p95 18.3 ms, 0 of 96 requests failed, hardware ceiling reached 18%

## Requests

- 96 requests: 96 succeeded, 0 failed
- outcomes: success 96
- NOTE: 96 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 16.31 req/s
- output throughput: 1097.9 tokens/s
- total throughput (prompt + output): 2978.6 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 16.31 req/s
- requests meeting the SLO: 100%
- TTFT p50: 52.2 ms
- TTFT p95: 199.2 ms
- TPOT p50: 17.5 ms
- TPOT p95: 18.3 ms
- end-to-end p95: 2309.8 ms
- start-up wave (first 32 requests, all starting together, outside the measured window): TTFT p50 143 ms, p95 290 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 0.3%
- highest sampled running requests (1 s polls): 31
- highest sampled waiting requests (1 s polls): 0
- preemptions: 0
- prefix-cache hits: 0 tokens
- prefix-cache hit rate: not measured
- KV cache capacity (measured by the engine): 1536192 tokens, 187.5 full-length requests at once; estimated before the run: 1371870 tokens (error -11%)

Tensward adds to every launch: --generation-config vllm (sampling follows the declared workload, not the checkpoint's generation_config.json), and the host, port, served model name and API key

## Engine defaults in effect (settings this setup leaves unset)

- async_scheduling: on, unless the setup is incompatible with it

## Hardware ceilings (theoretical upper bounds, not targets)

- GPU: L4
- memory bandwidth: 300.0 GB/s (bandwidth: datasheet; device-derived cross-check 300.0 GB/s)
- dense tensor rate: 121 TFLOPS (datasheet)
- per decode step (one sequence): 0.99 GB of weights (the vision encoder and the embedding gather are not read)
- KV cache read per sequence at the average context: 1.0 MiB
- average context per sequence: 83 tokens
- average running batch: 23.7 sequences
- decode ceiling, one sequence: 303 tok/s
- decode ceiling at the measured batch: 7,014 tok/s
- decode measured: 1,213 tok/s
- prefill ceiling: 169,043 tok/s
- prefill measured: 1,583 tok/s
- decode, share of its ceiling: 17.3%
- prefill, share of its ceiling: 0.9%
- window time the tokens need at both ceilings: 18.2%
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Tool calling (quality, not performance)

- requests that offered tools: 18
- produced a tool call: 100%
- of the 18 that did, arguments parse as JSON: 100%
- of the 18 that did, tool is one of the offered tools: 100%
- of the 18 that did, arguments satisfy the tool's required keys and property types: 100%
- caveat: counts every request that offered tools; prompts that should not call a tool count as misses
## GPU timeline (profiled - diagnostic only)

- profiled timing is not a speed claim: the profiler slows the engine, and this ran on a separate short launch after the clean measurement
- window: 3,068.6 ms, GPU busy 16.8%, 35,480 kernels
- 18,295 idle gaps of at least 50 us with no GPU activity: 71.4% of the window


## Compared with your current setup (run 20261004T170036Z-6648)

**Answers unchanged within noise**

Not identical: 6 of 10 prompts differ (first difference at character 27 in chat-05).
Your current setup reproduced its own answers for 5 of 10 prompts.
Put `VLLM_BATCH_INVARIANT=1` (vLLM's batch-invariant mode: beta, compute capability 8.0 or higher, slower, and it does not support prefix caching yet, vLLM issue #27433) and `--no-enable-prefix-caching` (vLLM enables prefix caching by default) in your `--current` command, for example `VLLM_BATCH_INVARIANT=1 vllm serve … --no-enable-prefix-caching`. Run `tensward init` again, because that changes the registration, then analyse your current setup again.

|  | current setup | this run | change |
|---|---|---|---|
| requests per second | 42.0 | 16.3 | -61% |
| output tokens per second | 2,828 | 1,098 | -61% |
| TTFT p50 (ms) | 51.8 | 52.2 | +1% |
| TTFT p95 (ms) | 120.3 | 199.2 | +66% |
| TPOT p95 (ms) | 7.7 | 18.3 | +139% |
| KV cache capacity (tokens) | 1,510,544 | 1,536,192 | +2% |

|  | current setup | this run |
|---|---|---|
| word-overlap similarity | 0.91 (its own repeats) | 0.90 (to your current setup) |
| answers with the same words | — | 62% |
| match with the reference | 0.72 | 0.72 |
| failed requests | 0% | 0% |
| answers cut short | 31% | 31% |
| empty answers | 0% | 0% |
| tool calls: produced a tool call | 100% | 100% |
| tool calls: valid JSON arguments | 100% | 100% |
| tool calls: known tool | 100% | 100% |
| tool calls: arguments match the schema | 100% | 100% |

A prompt counts as changed when its answers are more than 0.15 less similar to your current setup's than your current setup's are to each other (word overlap, at most 16 answers per prompt; a single changed number barely moves it, so read the answers below). Temperature: 0.

### Prompt `chat-05`

Your current setup:

```
Day 1: Lisbon, Portugal

- **Morning:** Start your day with a visit to the historic Alfama district, a UNESCO World Heritage site, which is a must-see. This area is known for its narrow streets, colorful buildings, and the famous Alfama Bridge, which is a UNESCO World Heritage site. You can also explore the Mercado do Alfama, a market that offers a variety of local products and food.

- **Afternoo… (trimmed)
```

This run:

```
Day 1: Lisbon, Portugal

- Morning: Start your day with a visit to the historic Alfama district, a UNESCO World Heritage site, which is a must-see for food lovers. The Alfama is a vibrant area with a mix of traditional Portuguese architecture and modern design. You can enjoy a traditional Portuguese breakfast at a local café or a delicious lunch at a local restaurant. Don't forget to try the local… (trimmed)
```

### Prompt `chat-06`

Your current setup:

```
This SQL query is designed to identify customers who have placed more than 5 orders within the last 30 days. The query uses a combination of `GROUP BY` and `HAVING` clauses to filter the results. Here's a breakdown of what the query does:

1. **GROUP BY customer_id**: This clause groups the results by the `customer_id` column, allowing the query to count the number of orders for each customer.

2.… (trimmed)
```

This run:

```
This SQL query is designed to identify customers who have placed more than 5 orders within the last 30 days. It groups orders by customer ID and counts the number of orders for each customer. The `HAVING` clause filters the results to only include customers who have placed more than 5 orders.

### How to Make It Faster

1. **Indexing**: Ensure that the `created_at` column is indexed. This will all… (trimmed)
```

### Prompt `chat-04`

Your current setup:

```
Voici la traduction en français :

"L'assemblée a été mises à jour à l'heure de jeudi à huit heures, veuillez confirmer que vous pouvez participer."
```

This run:

```
Voici la traduction en français :

"L'assemblée a été mises à jour à l'heure de jeudi à huit heures, veuillez confirmer que vous pouvez participer."
```

All prompts, with every distinct answer: `compare/20261004T170036Z-6648-vs-20261004T170351Z-d17c/answers.md`

Found something worth sharing, or a suggestion that was wrong? Tell us: https://github.com/Tensward/tensward/issues
