<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 1, L4: same workload after the suggested change

How this was produced (2026-10-01):

- run id: `20261001T064412Z-8fa0`
- GPU: NVIDIA L4 (24 GB)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.0 (development build). The report does not record the version; this is the version in the source tree at the time of the run, which is before the 0.1.1 release, so the report format is older than the current one
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0)
- `--current` command: `docker run --gpus all -p 8000:8000 -v ~/models:/models vllm/vllm-openai:v0.30.0 --model /models/qwen7b-awq --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes`
- analyse command: `tensward analyse --project ~/projA --runtime docker --engine-arg max-num-seqs=32`

This report is also published as [`examples/report-l4.md`](../../report-l4.md) or [`examples/report-l4-suggested.md`](../../report-l4-suggested.md); the content below is the same.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261001T064412Z-8fa0

## Your current setup + overrides (max-num-seqs=32)

- source: imported from --current
- settings: max_concurrent_requests=32, max_context_len=8192, kv_memory_fraction=0.9, tool_calling=True, tool_parser=hermes
- note: ignored docker options (Tensward runs it): -p
- note: your --current command wins; these serving fields of the configuration are ignored: max_concurrent_requests (configuration 8; your command sets 16), kv_memory_fraction (configuration 0.85; your command sets 0.9), prefill_batch_tokens (configuration 2048; your command does not set it), prefix_caching (configuration False; your command does not set it)
- measured: output 1198.9 tok/s, total 2950.8 tok/s, requests 14.75 req/s, goodput 14.75 req/s, TTFT p95 372 ms, TPOT p95 23.1 ms, 0 of 160 requests failed, hardware ceiling reached 75%

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 14.75 req/s
- output throughput: 1198.9 tokens/s
- total throughput (prompt + output): 2950.8 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 14.75 req/s
- requests meeting the SLO: 100%
- TTFT p50: 79.5 ms
- TTFT p95: 372.5 ms
- TPOT p50: 22.6 ms
- TPOT p95: 23.1 ms
- end-to-end p95: 3197.8 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 1.6%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 0
- preemptions: 0
- prefix-cache hits: 16672 tokens
- prefix-cache hit rate: 87.7%

Tensward adds to every launch: --generation-config vllm (sampling follows the declared workload, not the checkpoint's generation_config.json), and the host, port, served model name and API key

## Engine defaults in effect (settings this setup leaves unset)

- prefix_caching: on (for generative models)
- prefill_batch_tokens: 2048 (8192 on GPUs with 70 GiB or more that are not A100, 16384 from 160 GiB), with chunked prefill on
- async_scheduling: on, unless the setup is incompatible with it
- cuda_graphs: on

## Hardware ceilings (theoretical upper bounds, not targets)

- GPU: L4
- memory bandwidth: 300.0 GB/s (bandwidth: datasheet; device-derived cross-check 300.0 GB/s)
- dense tensor rate: 121 TFLOPS (datasheet)
- per decode step: 4.48 GB of weights, 56 KiB of KV cache per context token
- average context per sequence: 159 tokens
- average running batch: 26.0 sequences
- decode ceiling, one sequence: 67 tok/s
- decode ceiling at the measured batch: 1,653 tok/s
- decode measured: 1,199 tok/s
- prefill ceiling: 9,272 tok/s
- prefill measured: 215 tok/s
- decode, share of its ceiling: 72.5%
- prefill, share of its ceiling: 2.3%
- window time the tokens need at both ceilings: 74.8%
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
