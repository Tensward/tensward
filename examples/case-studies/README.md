# Case studies

Real before/after runs of Tensward on rented GPUs. Each case links the unedited `report.md`
files that Tensward wrote, with a provenance header giving the date, GPU, engine image, model,
inputs and commands. Every number below is copied from those files.

Rules for this page:

- Only runs whose inputs ship in this repository are included (`examples/config.json` with
  `examples/prompts.jsonl`, or `examples/config-images.json` with `examples/prompts-images.jsonl`).
- Each report was produced by the Tensward version named in the table. Older versions have older
  report formats, so section names differ between cases.
- One run per configuration. A single run on one machine is an example, not a promise.
- Reports are scrubbed of home paths only (`/home/ubuntu/` became `~/`). The headers say so.

All runs: engine `vllm/vllm-openai:v0.30.0` in Docker, 160 requests, 32 concurrent clients,
temperature 0. "Before" is the setup Tensward was given with `--current`; "after" is that setup
plus the one change listed. Quality column: Tensward compares answers only from 0.1.6 on.

| # | Model | GPU | Tensward | Change applied | Output tok/s | Requests/s | TTFT p95 | TPOT p95 | Answer quality | Reports |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen2.5-7B-Instruct-AWQ | NVIDIA L4 | 0.1.0 (dev build) | `max-num-seqs` 16 to 32 | 724.7 to 1198.9 | 8.91 to 14.75 | 2111 to 372 ms | 20.3 to 23.1 ms | not measured | [folder](l4-qwen7b-concurrency/) |
| 2 | Gemma 4 26B-A4B AWQ (images + tools) | NVIDIA A10G | 0.1.4 | enable tool calling (and a media limit) | 849.0 to 862.6 | 10.24 to 10.72 | 2072 to 1803 ms | 29.7 to 30.2 ms | not measured; 13 of 160 requests failed before, 0 after; 0% tool calls after | [folder](a10g-gemma4-tool-calling/) |
| 3 | Qwen2.5-7B-Instruct-AWQ | NVIDIA A10G | 0.1.6 | `kv-cache-dtype=fp8` | 1986.9 to 1550.5 | 24.44 to 20.50 | 325.2 to 324.7 ms | 14.0 to 15.8 ms | **changed 10 of 10 prompts; garbled** | [folder](a10g-qwen7b-fp8-kv-cache/) |
| 4 | Qwen2.5-7B-Instruct-AWQ | NVIDIA A10G | 0.1.6 | `enable-prefix-caching` | 1986.9 to 1985.5 | 24.44 to 24.42 | 325.2 to 326 ms | 14.0 to 14.1 ms | identical (answers unchanged) | [folder](a10g-qwen7b-prefix-caching/) |
| 5 | Qwen2.5-7B-Instruct-AWQ | NVIDIA A10G | 0.1.6 | `VLLM_BATCH_INVARIANT=1` | 1986.9 to 331.3 | 24.44 to 4.07 | 325.2 to 1058 ms | 14.0 to 85.1 ms | not compared with the baseline (see case) | [folder](a10g-qwen7b-batch-invariant/) |

Cases 3, 4 and 5 share one baseline: run `20261001T210738Z-cfc9`, stored in
[`a10g-qwen7b-fp8-kv-cache/baseline-report.md`](a10g-qwen7b-fp8-kv-cache/baseline-report.md).

## Case 1: more concurrent sequences on an L4

Tensward's baseline report showed running requests at the cap of 16 with 16 more waiting, KV-cache
usage at only 0.9%, and 32 concurrent clients declared in the workload. It suggested raising
`max-num-seqs` to 32. Applying only that change took output from 724.7 to 1198.9 tok/s, requests
from 8.91 to 14.75 req/s and TTFT p95 from 2111 ms to 372 ms. Goodput (requests inside TTFT 1000 ms
and TPOT 100 ms) went from 1.11 to 14.75 req/s. The cost: TPOT p95 rose from 20.3 to 23.1 ms,
because more sequences share each decode step. Answer quality was not measured in this run (the
feature arrived in 0.1.6). Reports:
[baseline](l4-qwen7b-concurrency/baseline-report.md),
[after](l4-qwen7b-concurrency/suggested-report.md).

## Case 2: a Gemma 4 setup that was refusing requests

On an A10G the baseline report said 13 of 160 requests failed, all with HTTP 400: "auto" tool
choice requires `--enable-auto-tool-choice` and `--tool-call-parser`. Tensward made two
suggestions. Enabling tool calling fixed the failures (0 of 160 failed afterwards). The other
suggestion, limiting media inputs to one image and no video, did not help: with it alone, output
moved from 849.0 to 851.5 tok/s, KV capacity from 13,839 to 13,797 tokens, and TTFT p95 stayed at
about 2070 ms ([report](a10g-gemma4-tool-calling/limit-media-only-report.md)); that suggestion
was removed from Tensward before release. With both changes: output 862.6 tok/s, TTFT p95
1803 ms. Read these numbers with care: the 13 requests that failed before are now served, so the
two runs do not do the same work. Also, tool calls did not work after the fix: the report shows 0%
of the 13 tool requests produced a tool call. The model answered in text that the example image
had no legible text ([recorded answer](a10g-gemma4-tool-calling/invoice-03-answer.md)); the
example images were redrawn after this run. Reports:
[baseline](a10g-gemma4-tool-calling/baseline-report.md),
[media limit only](a10g-gemma4-tool-calling/limit-media-only-report.md),
[tool calling](a10g-gemma4-tool-calling/tool-calling-report.md).

The hardware-ceiling lines in these three reports come from Tensward 0.1.4, which computed the mixture-of-experts decode ceiling assuming uniform routing. That is the slowest case, not an upper bound. Tensward 0.2.0 corrects this: the ceiling now assumes the fewest experts a step can read. The reports are left as Tensward wrote them.

## Case 3: FP8 KV cache, caught by the answer comparison

This is the case for comparing answers, not only speed. Setting `kv-cache-dtype=fp8` roughly
doubled the KV-cache capacity (248,000 to 483,632 tokens) and left latency percentiles about where
they were (TTFT p95 325.2 to 324.7 ms), so a capacity and latency check alone would not have flagged
it. In this run it did not even raise throughput: output fell from 1986.9 to 1550.5 tok/s (-22%)
and TPOT p95 rose from 14.0 to 15.8 ms. The comparison reported "Answers changed: 10 of 10
prompts". The answers were broken: tool-call prompts produced an empty answer or a tool call
wrapped in garbage with a leaked `<|im_start|>` token; a French translation degraded into
"au je jeudi à d dix heures,,"; SQL answers collapsed into repeated fragments. Tool calls produced
fell from 100% to 0%, empty answers rose from 0% to 10%, and word-overlap similarity to the current
setup was 0.30 where the current setup's own repeats scored 1.00. The best prompt (rag-01) still
scored only 0.82 and showed stutters. The cause was not investigated (candidates: no FP8 hardware
on an A10G, or uncalibrated KV scales for this model); the case shows only that this setup, on this
model and GPU, is not safe. Reports:
[baseline](a10g-qwen7b-fp8-kv-cache/baseline-report.md),
[FP8](a10g-qwen7b-fp8-kv-cache/fp8-report.md),
[answers excerpt](a10g-qwen7b-fp8-kv-cache/answers-excerpt.md).

## Case 4: prefix caching, no effect and no harm

Tensward suggested prefix caching because 2 of 10 prompts share a long prefix (the suggestion
appears on the batch-invariant setup, case 5, where prefix caching was off). On the baseline it did
nothing: vLLM v0.30.0 already has prefix caching on for this model, as the baseline report's
"engine defaults in effect" section lists. Output was 1986.9 vs 1985.5 tok/s and TTFT p95 325.2 vs
326 ms. The comparison reported all 10 prompts identical, so the change is safe but pointless
here. Reports: [baseline](a10g-qwen7b-fp8-kv-cache/baseline-report.md),
[prefix caching](a10g-qwen7b-prefix-caching/prefix-caching-report.md),
[answers excerpt](a10g-qwen7b-prefix-caching/answers-excerpt.md).

## Case 5: batch-invariant mode costs most of the throughput

`VLLM_BATCH_INVARIANT=1` (with prefix caching off) makes results independent of batching. On the
same workload and the same model files, output fell from 1986.9 to 331.3 tok/s (-83%), requests
from 24.44 to 4.07 req/s, TTFT p95 rose from 325.2 to 1058 ms and TPOT p95 from 14.0 to 85.1 ms.
The engine's log does not mention the mode, so the slowdown is easy to miss. Tensward did not
compare answers between this setup and the baseline (they are separate projects). A further run
with `max-num-seqs=8` added made it worse: 103.9 tok/s, TTFT p95 19836 ms, 24 requests waiting
while KV-cache use was 0.7%; the report's own suggestion was to raise concurrency to 32. The answers
of those two batch-invariant runs matched each other (10 of 10 unchanged). Reports:
[batch-invariant](a10g-qwen7b-batch-invariant/batch-invariant-report.md),
[with max-num-seqs=8](a10g-qwen7b-batch-invariant/max-num-seqs-8-report.md),
[answers excerpt](a10g-qwen7b-batch-invariant/answers-excerpt.md).
