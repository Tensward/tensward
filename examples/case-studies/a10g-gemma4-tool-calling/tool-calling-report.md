<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 2, Gemma 4 on an A10G: with tool calling enabled (and the media limit)

How this was produced (2026-10-01):

- run id: `20261001T161639Z-3fa8`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.4 (release validation run)
- model: Gemma 4 26B-A4B, AWQ int4 (compressed-tensors, group 32)
- inputs: [`examples/config-images.json`](../../config-images.json) and [`examples/prompts-images.jsonl`](../../prompts-images.jsonl) (160 requests over 12 prompts, 134 of them with one image each, 32 concurrent clients, temperature 0). The example images were redrawn afterwards so they carry legible text; this run used the earlier images.
- `--current` command: `vllm serve /m/gemma --max-model-len 8192` (run by Tensward in Docker with `--tool-call-parser gemma4`, which Tensward took from the checkpoint)
- analyse command: `tensward analyse --project ~/p --runtime docker --engine-arg 'limit-mm-per-prompt={"image": 1, "video": 0}' --engine-arg enable-auto-tool-choice`

The command shown is reconstructed from the `Try:` lines of the previous report and the run's recorded engine arguments; the exact shell line is not stored with the run.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261001T161639Z-3fa8

## Your current setup + overrides (limit-mm-per-prompt={"image": 1, "video": 0}, enable-auto-tool-choice)

- source: imported from --current
- settings: max_context_len=8192, tool_calling=True, tool_parser=gemma4, media_limits={'image': 1, 'video': 0}
- note: your --current command wins; these serving fields of the configuration are ignored: max_concurrent_requests (configuration 8; your command does not set it), kv_memory_fraction (configuration 0.85; your command does not set it), prefill_batch_tokens (configuration 2048; your command does not set it), prefix_caching (configuration False; your command does not set it)
- measured: output 862.6 tok/s, total 4104.1 tok/s, requests 10.72 req/s, goodput 8.64 req/s, TTFT p95 1803 ms, TPOT p95 30.2 ms, 0 of 160 requests failed, decode at 85.2% of its ceiling

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 12 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 10.72 req/s
- output throughput: 862.6 tokens/s
- total throughput (prompt + output): 4104.1 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 8.64 req/s
- requests meeting the SLO: 81%
- TTFT p50: 94.0 ms
- TTFT p95: 1803.1 ms
- TPOT p50: 29.6 ms
- TPOT p95: 30.2 ms
- end-to-end p95: 5130.5 ms

## Requests with and without images

- with images: 134 requests, TTFT p50 95 ms, TTFT p95 1803 ms, TPOT p95 30 ms, mean prompt tokens 353
- without images: 26 requests, TTFT p50 90 ms, TTFT p95 1798 ms, TPOT p95 30 ms, mean prompt tokens 41
- images per request that carries images: 1.0
- the prompt tokens of a request with images include the image tokens the engine counted

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 56.1%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 1
- preemptions: 0
- prefix-cache hits: 43424 tokens
- prefix-cache hit rate: 89.8%
- KV cache capacity (measured by the engine): 13797 tokens, 1.7 full-length requests at once; estimated before the run: 15788 tokens (error +14%)

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
- per decode step (one sequence): 4.00 GB of weights (the vision encoder and the embedding gather are not read)
- KV cache read per sequence at the average context: 73.7 MiB
- mixture of experts: 8 of 128 experts per token
- at the measured batch a decode step reads about 105 of 128 experts per layer (10.55 GB), assuming uniform routing; real routing is skewed and reads fewer. Steps that mix prefill read up to all experts, so the decode ceiling holds for decode-only steps
- average context per sequence: 343 tokens
- average running batch: 26.7 sequences
- decode ceiling, one sequence: 147 tok/s
- decode ceiling at the measured batch: 1,013 tok/s
- decode measured: 863 tok/s
- prefill ceiling: not measured
- prefill measured: 332 tok/s
- decode, share of its ceiling: 85.2%
- prefill, share of its ceiling: not measured
- window time the tokens need at both ceilings: not measured
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Quantization

- checkpoint: compressed-tensors int4 group 32
- kernel: MarlinLinearKernel (CompressedTensorsWNA16), MARLIN (WNA16 MoE)

## Tool calling (quality, not performance)

- requests that offered tools: 13
- produced a tool call: 0%
- of the 0 that did, arguments parse as JSON: no tool call to judge
- of the 0 that did, tool is one of the offered tools: no tool call to judge
- of the 0 that did, arguments satisfy the tool's required keys and property types: no tool call to judge
- caveat: counts every request that offered tools; prompts that should not call a tool count as misses
## Suggested experiments

No experiment is suggested by the measured signals.
