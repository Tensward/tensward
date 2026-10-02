<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 3, FP8 KV cache: baseline (your current setup)

How this was produced (2026-10-01):

- run id: `20261001T210738Z-cfc9`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.6 (release validation run)
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0)
- `--current` command: `vllm serve /m/qwen --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes`
- analyse command: `tensward analyse --project ~/p --runtime docker`

The command shown is reconstructed from the run's recorded engine arguments and the report's own `Try:` lines; the exact shell line is not stored with the run.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261001T210738Z-cfc9

## Your current setup

- source: imported from --current
- settings: max_context_len=8192, tool_calling=True, tool_parser=hermes
- note: your --current command wins; these serving fields of the configuration are ignored: max_concurrent_requests (configuration 8; your command does not set it), kv_memory_fraction (configuration 0.85; your command does not set it), prefill_batch_tokens (configuration 2048; your command does not set it), prefix_caching (configuration False; your command does not set it)
- measured: output 1986.9 tok/s, total 4890.3 tok/s, requests 24.44 req/s, goodput 24.44 req/s, TTFT p95 325 ms, TPOT p95 14.0 ms, 0 of 160 requests failed, decode at 68.3% of its ceiling

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 24.44 req/s
- output throughput: 1986.9 tokens/s
- total throughput (prompt + output): 4890.3 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 24.44 req/s
- requests meeting the SLO: 100%
- TTFT p50: 62.2 ms
- TTFT p95: 325.2 ms
- TPOT p50: 13.4 ms
- TPOT p95: 14.0 ms
- end-to-end p95: 1986.0 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 1.2%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 0
- preemptions: 0
- prefix-cache hits: 16672 tokens
- prefix-cache hit rate: 87.7%
- KV cache capacity (measured by the engine): 248000 tokens, 30.3 full-length requests at once; estimated before the run: 243439 tokens (error -2%)

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
- KV cache read per sequence at the average context: 8.7 MiB
- average context per sequence: 159 tokens
- average running batch: 22.7 sequences
- decode ceiling, one sequence: 134 tok/s
- decode ceiling at the measured batch: 2,908 tok/s
- decode measured: 1,987 tok/s
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
