<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 5, batch-invariant mode: batch-invariant mode (`VLLM_BATCH_INVARIANT=1`)

How this was produced (2026-10-01):

- run id: `20261001T211447Z-06a6`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.6 (release validation run)
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0)
- `--current` command: the same command as case 3 with `VLLM_BATCH_INVARIANT=1` set in the environment and prefix caching off
- analyse command: `tensward analyse --project ~/pbi --runtime docker`

This run used a second project (`~/pbi`) over the same model files (identical weight fingerprint) and the same workload. Its comparison baseline is `../a10g-qwen7b-fp8-kv-cache/baseline-report.md` (run `20261001T210738Z-cfc9`); Tensward did not compare answers across the two projects. The command shown is reconstructed from the run's recorded engine arguments and the report's own `Try:` lines; the exact shell line is not stored with the run.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261001T211447Z-06a6

## Your current setup

- source: imported from --current
- settings: max_context_len=8192, prefix_caching=False, tool_calling=True, tool_parser=hermes, VLLM_BATCH_INVARIANT=1
- note: your --current command wins; these serving fields of the configuration are ignored: max_concurrent_requests (configuration 8; your command does not set it), kv_memory_fraction (configuration 0.85; your command does not set it), prefill_batch_tokens (configuration 2048; your command does not set it)
- measured: output 331.3 tok/s, total 815.3 tok/s, requests 4.07 req/s, goodput 3.72 req/s, TTFT p95 1058 ms, TPOT p95 85.1 ms, 0 of 160 requests failed, decode at 9.7% of its ceiling

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 4.07 req/s
- output throughput: 331.3 tokens/s
- total throughput (prompt + output): 815.3 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 3.72 req/s
- requests meeting the SLO: 91%
- TTFT p50: 389.9 ms
- TTFT p95: 1057.8 ms
- TPOT p50: 82.0 ms
- TPOT p95: 85.1 ms
- end-to-end p95: 11027.6 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 2.4%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 0
- preemptions: 0
- prefix-cache hits: 0 tokens
- prefix-cache hit rate: not measured
- KV cache capacity (measured by the engine): 218544 tokens, 26.7 full-length requests at once; estimated before the run: 243439 tokens (error +11%)

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
- average running batch: 27.0 sequences
- decode ceiling, one sequence: 134 tok/s
- decode ceiling at the measured batch: 3,425 tok/s
- decode measured: 331 tok/s
- prefill ceiling: not measured
- prefill measured: 484 tok/s
- decode, share of its ceiling: 9.7%
- prefill, share of its ceiling: not measured
- window time the tokens need at both ceilings: not measured
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Quantization

- checkpoint: awq int4 group 128
- kernel: not detected in the server log

## Tool calling (quality, not performance)

- requests that offered tools: 32
- produced a tool call: 100%
- of the 32 that did, arguments parse as JSON: 100%
- of the 32 that did, tool is one of the offered tools: 100%
- of the 32 that did, arguments satisfy the tool's required keys and property types: 100%
- caveat: counts every request that offered tools; prompts that should not call a tool count as misses
## Suggested experiments

- `prefix-caching`: 2 of 10 prompts (20%) share a prompt prefix of up to 114 words with another prompt
  Try: `tensward analyse --project ~/pbi --engine-arg enable-prefix-caching`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.
