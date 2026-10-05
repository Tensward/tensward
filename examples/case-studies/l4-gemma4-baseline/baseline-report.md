<!-- Provenance. This is the genuine, unedited `report.md` of the run, apart from this header and one scrub: the box path `/home/ubuntu/p/<project>` (the project directory on the rented machine) was replaced by `<project>`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, usernames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 9, Gemma 4 26B-A4B AWQ on an L4: baseline

How this was produced (2026-10-04):

- run id: `20261004T171843Z-7dc7`
- GPU: NVIDIA L4 (24 GB), AWS g6.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.3.1 (release candidate, release validation run).
- model: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (compressed-tensors int4)
- inputs: [`config.json`](config.json) (int4, bfloat16, max-model-len 4096, 160 requests, 32 concurrent clients, max-num-seqs 8) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (10 prompts, temperature 0, seed 0)
- init command: `tensward init --project <project> --model <local copy of cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit> --config config.json --prompts examples/prompts.jsonl`
- analyse command: `tensward analyse --project <project>`

Produced by the Tensward 0.3.1 release candidate. The released 0.3.1 words two Diagnosis lines differently: classes whose threshold is calibrated are listed under "not crossed:" apart from the uncalibrated ones, and a run with no bottleneck says "some thresholds not yet calibrated". Numbers and diagnoses are identical.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

Correction (Tensward 0.3.2): the throughput figures in this report were measured over a span that included the start-up wave's tail and the drain after the last request. Recomputed from the same recorded run over the steady window of 47.6 s: output 182.8 to 219.9 tok/s, requests 2.16 to 2.63 req/s. TTFT and TPOT percentiles are unchanged. The report below is left as Tensward 0.3.1 wrote it.

---

# Tensward analysis 20261004T171843Z-7dc7

Ran: vLLM 0.30.0 (docker image vllm/vllm-openai:v0.30.0) on NVIDIA L4 (driver 595.91.07, CUDA 13.2); checkpoint: safetensors, compressed-tensors int4 group 32

## Diagnosis

Bottleneck: queueing before scheduling (confidence: high)
- evidence: requests spent 99% of their time to first token queued; 32 requests were in flight against a concurrency cap of 8
- not crossed (uncalibrated thresholds): KV-cache capacity, prefill compute, prefill stalling decode, decode memory bandwidth, attention / long context
- speculation: off or not reported
- can't tell here:
  - GPU compute, tensor-bound kernels — kernel counters: run with `--counters`
  - host / CPU overhead — a GPU trace: run with `--trace`
- not modelled: offload / PCIe (needs offload signals, which come with the llama.cpp engine); multi-GPU communication (Tensward measures one GPU)
- fit and failed requests: every request was served
- workload shape: 1.1 prompt tokens computed per generated token

## What to try next

For queueing before scheduling:
- `raise-concurrency`: running requests hit max_concurrent_requests 8 with 24 waiting, and the highest sampled KV-cache usage is only 23.3%; at your declared load of 32 concurrent clients, up to 32 requests were in flight, so 16 is worth trying (the load is what your configuration declares, not measured traffic). It usually cuts queueing (TTFT) but slows each token (TPOT) as more sequences share every step - measure both
  evidence: strong; may cost: TPOT rises as more sequences share each step; more KV cache in use
  Try: `tensward analyse --project gemma --engine-arg max-num-seqs=16`
Other changes the measurements support:
- `prefix-caching`: 2 of 10 prompts (20%) share a prompt prefix of up to 114 words with another prompt
  evidence: strong; may cost: a little GPU memory for the cache
  Try: `tensward analyse --project gemma --engine-arg enable-prefix-caching`
- `no-media-encoders`: the checkpoint takes images and video but no prompt sends any; serving only its text model stops the engine reserving memory for the media encoders and lifts the minimum batch size they impose (requests with images would then be rejected)
  evidence: ours; may cost: requests with images are rejected
  Try: `tensward analyse --project gemma --engine-arg language-model-only`

To serve a change, give `tensward serve start` the same `--project` and `--engine-arg` options.

## Your current setup

- source: declared in config
- settings: dtype=bfloat16, max_concurrent_requests=8, max_context_len=4096, kv_memory_fraction=0.85, prefill_batch_tokens=4096, prefix_caching=False, tool_calling=True, tool_parser=gemma4
- measured: output 182.8 tok/s, total 387.8 tok/s, requests 2.16 req/s, goodput 0.00 req/s, TTFT p95 10282 ms, TPOT p95 36.3 ms, 0 of 160 requests failed, hardware ceiling reached 41%

## Requests

- 160 requests: 160 succeeded, 0 failed
- outcomes: success 160
- NOTE: 160 requests from 10 distinct prompts: prefix-cache hit rate and similar signals are inflated by repetition; use more distinct prompts for realistic numbers

## Performance (client side)

- request throughput: 2.16 req/s
- output throughput: 182.8 tokens/s
- total throughput (prompt + output): 387.8 tokens/s
- goodput (TTFT <= 1000 ms and TPOT <= 100 ms per request): 0.00 req/s
- requests meeting the SLO: 0%
- TTFT p50: 9483.5 ms
- TTFT p95: 10282.5 ms
- TPOT p50: 35.6 ms
- TPOT p95: 36.3 ms
- end-to-end p95: 14658.3 ms
- start-up wave (first 32 requests, all starting together, outside the measured window): TTFT p50 4725 ms, p95 10168 ms

## Engine signals (from the engine's metrics)

- highest sampled KV-cache usage (1 s polls): 23.3%
- highest sampled running requests (1 s polls): 8
- highest sampled waiting requests (1 s polls): 24
- preemptions: 0
- prefix-cache hits: 0 tokens
- prefix-cache hit rate: not measured
- KV cache capacity (measured by the engine): 5565 tokens, 1.4 full-length requests at once; estimated before the run: 2925 tokens (error -47%)

Tensward adds to every launch: --generation-config vllm (sampling follows the declared workload, not the checkpoint's generation_config.json), and the host, port, served model name and API key

## Engine defaults in effect (settings this setup leaves unset)

- async_scheduling: on, unless the setup is incompatible with it
- cuda_graphs: on

## Hardware ceilings (theoretical upper bounds, not targets)

- GPU: L4
- memory bandwidth: 300.0 GB/s (bandwidth: datasheet; device-derived cross-check 300.0 GB/s)
- dense tensor rate: 121 TFLOPS (datasheet)
- per decode step (one sequence): 4.00 GB of weights (the vision encoder and the embedding gather are not read)
- KV cache read per sequence at the average context: 28.4 MiB
- mixture of experts: 8 of 128 experts per token
- at the measured batch a decode step reads about 50 of 128 experts per layer (5.04 GB), with uniform routing; real routing overlaps and reads fewer, so measured decode can sit between this and the ceiling, which assumes each layer reads only 8 experts. Steps that mix prefill read up to all experts, so the decode ceiling holds for decode-only steps
- average context per sequence: 132 tokens
- average running batch: 7.7 sequences
- decode ceiling, one sequence: 74 tok/s
- decode ceiling at the measured batch: 547 tok/s
- decode measured: 217 tok/s
- prefill ceiling: 19,615 tok/s
- prefill measured: 245 tok/s
- decode, share of its ceiling: 39.7%
- prefill, share of its ceiling: 1.2%
- window time the tokens need at both ceilings: 40.9%
- prefill and decode share the GPU, so each share alone understates how busy it was; the last line combines them. Attention FLOPs and activation traffic are ignored.

## Quantization

- checkpoint: compressed-tensors int4 group 32
- kernel: MarlinLinearKernel (CompressedTensorsWNA16), MARLIN (WNA16 MoE)

## Tool calling (quality, not performance)

- requests that offered tools: 32
- produced a tool call: 100%
- of the 32 that did, arguments parse as JSON: 100%
- of the 32 that did, tool is one of the offered tools: 100%
- of the 32 that did, arguments satisfy the tool's required keys and property types: 100%
- caveat: counts every request that offered tools; prompts that should not call a tool count as misses
Found something worth sharing, or a suggestion that was wrong? Tell us: https://github.com/Tensward/tensward/issues
