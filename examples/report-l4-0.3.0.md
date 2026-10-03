# Tensward analysis 20261003T203841Z-3604

Ran: vLLM 0.30.0 (docker image vllm/vllm-openai:v0.30.0) on NVIDIA L4 (driver 595.91.07, CUDA 13.2); checkpoint: safetensors, awq int4 group 128

## Diagnosis

Bottleneck: queueing before scheduling (confidence: likely; thresholds not yet calibrated on real GPUs)
- evidence: requests spent 98% of their time to first token queued; 32 requests were in flight against a concurrency cap of 8
- also seen: decode memory bandwidth (likely): decode ran at 72% of the memory-bandwidth ceiling at the measured batch
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
- `raise-concurrency`: running requests hit max_concurrent_requests 8 with 24 waiting, and the highest sampled KV-cache usage is only 0.6%; at your declared load of 32 concurrent clients, up to 32 requests were in flight, so 32 is worth trying (the load is what your configuration declares, not measured traffic). It usually cuts queueing (TTFT) but slows each token (TPOT) as more sequences share every step - measure both
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
- measured: output 356.2 tok/s, total 876.7 tok/s, requests 4.38 req/s, goodput 0.25 req/s, TTFT p95 5783 ms, TPOT p95 22.7 ms, 0 of 160 requests failed, hardware ceiling reached 78%

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 4.38 req/s
- output throughput: 356.2 tokens/s
- total throughput (prompt + output): 876.7 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 0.25 req/s
- requests meeting the SLO: 6%
- TTFT p50: 5414.3 ms
- TTFT p95: 5783.1 ms
- TPOT p50: 21.2 ms
- TPOT p95: 22.7 ms
- end-to-end p95: 8422.6 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 0.6%
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
- KV cache read per sequence at the average context: 8.7 MiB
- average context per sequence: 159 tokens
- average running batch: 7.5 sequences
- decode ceiling, one sequence: 67 tok/s
- decode ceiling at the measured batch: 494 tok/s
- decode measured: 356 tok/s
- prefill ceiling: 9,271 tok/s
- prefill measured: 521 tok/s
- decode, share of its ceiling: 72.1%
- prefill, share of its ceiling: 5.6%
- window time the tokens need at both ceilings: 77.8%
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
