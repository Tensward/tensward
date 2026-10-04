<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one edit: the box path `/home/ubuntu/p/<project>` (the project directory on the rented machine) was replaced by `<project>`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, usernames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 10, what changed: baseline

How this was produced (2026-10-04):

- run id: `20261004T181354Z-ec62`
- GPU: NVIDIA L4 (24 GB), AWS g6.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.3.1 (released version, release validation run).
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0, `max-num-seqs` 8 as declared in the config)
- init command: `tensward init --project <project> --model <local copy of Qwen2.5-7B-Instruct-AWQ> --config examples/config.json --prompts examples/prompts.jsonl`
- analyse command: `tensward analyse --project <project>`

The text below the line is unedited apart from this header and the path edit named above. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261004T181354Z-ec62

Ran: vLLM 0.30.0 (docker image vllm/vllm-openai:v0.30.0) on NVIDIA L4 (driver 595.91.07, CUDA 13.2); checkpoint: safetensors, awq int4 group 128

## Diagnosis

Bottleneck: queueing before scheduling (confidence: high)
- evidence: requests spent 99% of their time to first token queued; 32 requests were in flight against a concurrency cap of 8
- also seen: decode memory bandwidth (likely): decode ran at 69% of the memory-bandwidth ceiling at the measured batch
- not crossed (uncalibrated thresholds): KV-cache capacity, prefill compute, prefill stalling decode, attention / long context
- speculation: off or not reported
- can't tell here:
  - GPU compute, tensor-bound kernels — kernel counters: run with `--counters`
  - host / CPU overhead — a GPU trace: run with `--trace`
- not modelled: offload / PCIe (needs offload signals, which come with the llama.cpp engine); multi-GPU communication (Tensward measures one GPU)
- fit and failed requests: every request was served
- workload shape: 1.5 prompt tokens computed per generated token

## What to try next

For queueing before scheduling:
- `raise-concurrency`: running requests hit max_concurrent_requests 8 with 24 waiting, and the highest sampled KV-cache usage is only 0.7%; at your declared load of 32 concurrent clients, up to 32 requests were in flight, so 32 is worth trying (the load is what your configuration declares, not measured traffic). It usually cuts queueing (TTFT) but slows each token (TPOT) as more sequences share every step - measure both
  evidence: strong; may cost: TPOT rises as more sequences share each step; more KV cache in use
  Try: `tensward analyse --project <project> --engine-arg max-num-seqs=32`
For decode memory bandwidth:
- no change in this engine's playbook applies here
Other changes the measurements support:
- `prefix-caching`: 2 of 10 prompts (20%) share a prompt prefix of up to 114 words with another prompt
  evidence: strong; may cost: a little GPU memory for the cache
  Try: `tensward analyse --project <project> --engine-arg enable-prefix-caching`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.

## Your current setup

- source: declared in config
- settings: dtype=float16, max_concurrent_requests=8, max_context_len=8192, kv_memory_fraction=0.85, prefill_batch_tokens=2048, prefix_caching=False, tool_calling=True, tool_parser=hermes
- measured: output 298.6 tok/s, total 734.8 tok/s, requests 3.67 req/s, goodput 0.00 req/s, TTFT p95 5800 ms, TPOT p95 22.9 ms, 0 of 160 requests failed, hardware ceiling reached 74%

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 3.67 req/s
- output throughput: 298.6 tokens/s
- total throughput (prompt + output): 734.8 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 0.00 req/s
- requests meeting the SLO: 0%
- TTFT p50: 5501.6 ms
- TTFT p95: 5800.3 ms
- TPOT p50: 21.4 ms
- TPOT p95: 22.9 ms
- end-to-end p95: 8489.2 ms
- start-up wave (first 32 requests, all starting together, outside the measured window): TTFT p50 3189 ms, p95 6361 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 0.7%
- highest sampled running requests (1 s polls): 8
- highest sampled waiting requests (1 s polls): 24
- preemptions: 0
- prefix-cache hits: 0 tokens
- prefix-cache hit rate: not measured
- KV cache capacity (measured by the engine): 228432 tokens, 27.9 full-length requests at once; estimated before the run: 214056 tokens (error -6%)

Tensward adds to every launch: --generation-config vllm (sampling follows the declared workload, not the checkpoint's generation_config.json), and the host, port, served model name and API key

## Engine defaults in effect (settings this setup leaves unset)

- async_scheduling: on, unless the setup is incompatible with it
- cuda_graphs: on

## Hardware ceilings (theoretical upper bounds, not targets)

- GPU: L4
- memory bandwidth: 300.0 GB/s (bandwidth: datasheet; device-derived cross-check 300.0 GB/s)
- dense tensor rate: 121 TFLOPS (datasheet)
- per decode step (one sequence): 4.48 GB of weights (the vision encoder and the embedding gather are not read)
- KV cache read per sequence at the average context: 8.5 MiB
- average context per sequence: 155 tokens
- average running batch: 7.8 sequences
- decode ceiling, one sequence: 67 tok/s
- decode ceiling at the measured batch: 516 tok/s
- decode measured: 354 tok/s
- prefill ceiling: 9,271 tok/s
- prefill measured: 516 tok/s
- decode, share of its ceiling: 68.6%
- prefill, share of its ceiling: 5.6%
- window time the tokens need at both ceilings: 74.1%
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
Found something worth sharing, or a suggestion that was wrong? Tell us: https://github.com/Tensward/tensward/issues
