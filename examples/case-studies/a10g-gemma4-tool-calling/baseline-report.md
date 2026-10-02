<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 2, Gemma 4 on an A10G: baseline (your current setup)

How this was produced (2026-10-01):

- run id: `20261001T160126Z-9db1`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.4 (release validation run)
- model: Gemma 4 26B-A4B, AWQ int4 (compressed-tensors, group 32)
- inputs: [`examples/config-images.json`](../../config-images.json) and [`examples/prompts-images.jsonl`](../../prompts-images.jsonl) (160 requests over 12 prompts, 134 of them with one image each, 32 concurrent clients, temperature 0). The example images were redrawn afterwards so they carry legible text; this run used the earlier images.
- `--current` command: `vllm serve /m/gemma --max-model-len 8192` (run by Tensward in Docker with `--tool-call-parser gemma4`, which Tensward took from the checkpoint)
- analyse command: `tensward analyse --project ~/p --runtime docker`

The command shown is reconstructed from the `Try:` lines of the previous report and the run's recorded engine arguments; the exact shell line is not stored with the run.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Tensward analysis 20261001T160126Z-9db1

## Your current setup

- source: imported from --current
- settings: max_context_len=8192, tool_parser=gemma4
- note: your --current command wins; these serving fields of the configuration are ignored: max_concurrent_requests (configuration 8; your command does not set it), kv_memory_fraction (configuration 0.85; your command does not set it), prefill_batch_tokens (configuration 2048; your command does not set it), prefix_caching (configuration False; your command does not set it)
- measured: output 849.0 tok/s, total 3793.5 tok/s, requests 10.24 req/s, goodput 8.01 req/s, TTFT p95 2072 ms, TPOT p95 29.7 ms, 13 of 160 requests failed, decode at 90.9% of its ceiling
- **your current setup fails requests on this workload; fixing that is the first improvement**

## Requests

- 160 requests: 147 succeeded, 13 failed
- outcomes: error 13, success 147
- NOTE: 160 requests from 12 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

### Why requests failed

- 13 x HTTP 400: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set (prompts: invoice-03)

## Performance (client side)

- request throughput: 10.24 req/s
- output throughput: 849.0 tokens/s
- total throughput (prompt + output): 3793.5 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 8.01 req/s
- requests meeting the SLO: 72%
- TTFT p50: 94.6 ms
- TTFT p95: 2072.3 ms
- TPOT p50: 28.9 ms
- TPOT p95: 29.7 ms
- end-to-end p95: 5423.2 ms

## Requests with and without images

- with images: 134 requests, TTFT p50 95 ms, TTFT p95 2072 ms, TPOT p95 30 ms, mean prompt tokens 341
- without images: 26 requests, TTFT p50 92 ms, TTFT p95 2070 ms, TPOT p95 30 ms, mean prompt tokens 41
- images per request that carries images: 1.0
- the prompt tokens of a request with images include the image tokens the engine counted

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 54.7%
- highest sampled running requests (1 s polls): 32
- highest sampled waiting requests (1 s polls): 0
- preemptions: 0
- prefix-cache hits: 38048 tokens
- prefix-cache hit rate: 90.0%
- KV cache capacity (measured by the engine): 13839 tokens, 1.7 full-length requests at once; estimated before the run: 15788 tokens (error +14%)

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
- KV cache read per sequence at the average context: 70.7 MiB
- mixture of experts: 8 of 128 experts per token
- at the measured batch a decode step reads about 99 of 128 experts per layer (9.96 GB), assuming uniform routing; real routing is skewed and reads fewer. Steps that mix prefill read up to all experts, so the decode ceiling holds for decode-only steps
- average context per sequence: 329 tokens
- average running batch: 23.1 sequences
- decode ceiling, one sequence: 147 tok/s
- decode ceiling at the measured batch: 934 tok/s
- decode measured: 849 tok/s
- prefill ceiling: not measured
- prefill measured: 294 tok/s
- decode, share of its ceiling: 90.9%
- prefill, share of its ceiling: not measured
- window time the tokens need at both ceilings: not measured
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Quantization

- checkpoint: compressed-tensors int4 group 32
- kernel: MarlinLinearKernel (CompressedTensorsWNA16), MARLIN (WNA16 MoE)

## Checks

- tool calling: the workload offers tools but tool calling is not enabled, so the server refuses those requests; set `tool_calling: true` in the serving configuration's case (parser gemma4) or add `--engine-arg enable-auto-tool-choice`
## Suggested experiments

- `enable-tool-calling`: the workload offers tools but tool calling is off, so the server refuses them; the checkpoint's tool parser is gemma4
  Try: `tensward analyse --project ~/p --engine-arg enable-auto-tool-choice`
- `limit-media`: the workload sends at most 1 images per prompt and no video, but the engine reserves encoder memory and a minimum batch size for video; limiting the inputs to what the workload sends frees both
  Try: `tensward analyse --project ~/p --engine-arg 'limit-mm-per-prompt={"image": 1, "video": 0}'`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.
