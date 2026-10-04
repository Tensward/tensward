<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the box path `/home/ubuntu/p/<project>` (the project directory on the rented machine) was replaced by `<project>`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, usernames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 7, Qwen2.5-7B AWQ, n-gram speculation: baseline

How this was produced (2026-10-04):

- run id: `20261004T170652Z-a1dd`
- GPU: NVIDIA L4 (24 GB), AWS g6.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.3.1 (release candidate, release validation run).
- model: Qwen/Qwen2.5-7B-Instruct-AWQ
- inputs: [`config.json`](config.json) (AWQ int4, float16, 96 requests, 32 concurrent clients, max-num-seqs 32) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (10 prompts, temperature 0, seed 0)
- init command: `tensward init --project <project> --model <local copy of Qwen/Qwen2.5-7B-Instruct-AWQ> --config config.json --prompts examples/prompts.jsonl`
- analyse command: `tensward analyse --project <project>`

Produced by the Tensward 0.3.1 release candidate. The released 0.3.1 words two Diagnosis lines differently: classes whose threshold is calibrated are listed under "not crossed:" apart from the uncalibrated ones, and a run with no bottleneck says "some thresholds not yet calibrated". Numbers and diagnoses are identical.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261004T170652Z-a1dd

Ran: vLLM 0.30.0 (docker image vllm/vllm-openai:v0.30.0) on NVIDIA L4 (driver 595.91.07, CUDA 13.2); checkpoint: safetensors, awq int4 group 128

## Diagnosis

Bottleneck: decode memory bandwidth (confidence: likely)
- evidence: decode ran at 50% of the memory-bandwidth ceiling at the measured batch
- not crossed (uncalibrated thresholds): queueing before scheduling, KV-cache capacity, prefill compute, prefill stalling decode, attention / long context
- speculation: off or not reported
- can't tell here:
  - GPU compute, tensor-bound kernels — kernel counters: run with `--counters`
  - host / CPU overhead — a GPU trace: run with `--trace`
- not modelled: offload / PCIe (needs offload signals, which come with the llama.cpp engine); multi-GPU communication (Tensward measures one GPU)
- fit and failed requests: every request was served
- workload shape: 1.2 prompt tokens computed per generated token

## What to try next

For decode memory bandwidth:
- no change in this engine's playbook applies here
Other changes the measurements support:
- `prefix-caching`: 2 of 10 prompts (20%) share a prompt prefix of up to 114 words with another prompt
  evidence: strong; may cost: a little GPU memory for the cache
  Try: `tensward analyse --project ngram --engine-arg enable-prefix-caching`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.

## Your current setup

- source: declared in config
- settings: dtype=float16, max_concurrent_requests=32, max_context_len=8192, kv_memory_fraction=0.85, prefill_batch_tokens=2048, prefix_caching=False, tool_calling=True, tool_parser=hermes
- measured: output 690.7 tok/s, total 1648.6 tok/s, requests 8.31 req/s, goodput 8.31 req/s, TTFT p95 587 ms, TPOT p95 36.7 ms, 0 of 96 requests failed, hardware ceiling reached 61%

## Requests

- 96 requests: 96 succeeded, 0 failed
- outcomes: success 96
- NOTE: 96 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 8.31 req/s
- output throughput: 690.7 tokens/s
- total throughput (prompt + output): 1648.6 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 8.31 req/s
- requests meeting the SLO: 100%
- TTFT p50: 171.4 ms
- TTFT p95: 587.4 ms
- TPOT p50: 32.1 ms
- TPOT p95: 36.7 ms
- end-to-end p95: 4820.2 ms
- start-up wave (first 32 requests, all starting together, outside the measured window): TTFT p50 694 ms, p95 1347 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 2.2%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 0
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
- KV cache read per sequence at the average context: 6.3 MiB
- average context per sequence: 116 tokens
- average running batch: 23.8 sequences
- decode ceiling, one sequence: 67 tok/s
- decode ceiling at the measured batch: 1,538 tok/s
- decode measured: 773 tok/s
- prefill ceiling: 9,271 tok/s
- prefill measured: 959 tok/s
- decode, share of its ceiling: 50.3%
- prefill, share of its ceiling: 10.3%
- window time the tokens need at both ceilings: 60.6%
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Quantization

- checkpoint: awq int4 group 128
- kernel: MarlinLinearKernel (AutoAWQMarlinLinearMethod)

## Tool calling (quality, not performance)

- requests that offered tools: 18
- produced a tool call: 100%
- of the 18 that did, arguments parse as JSON: 100%
- of the 18 that did, tool is one of the offered tools: 100%
- of the 18 that did, arguments satisfy the tool's required keys and property types: 100%
- caveat: counts every request that offered tools; prompts that should not call a tool count as misses
Found something worth sharing, or a suggestion that was wrong? Tell us: https://github.com/Tensward/tensward/issues
