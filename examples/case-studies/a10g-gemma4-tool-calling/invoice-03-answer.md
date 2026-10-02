<!-- Provenance. This is the genuine, unedited one record of `responses.jsonl` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 2, Gemma 4 on an A10G: one recorded answer to the prompt that failed before

How this was produced (2026-10-01):

- run id: `20261001T161639Z-3fa8`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.4 (release validation run)
- model: Gemma 4 26B-A4B, AWQ int4 (compressed-tensors, group 32)
- inputs: [`examples/config-images.json`](../../config-images.json) and [`examples/prompts-images.jsonl`](../../prompts-images.jsonl) (160 requests over 12 prompts, 134 of them with one image each, 32 concurrent clients, temperature 0). The example images were redrawn afterwards so they carry legible text; this run used the earlier images.
- `--current` command: `vllm serve /m/gemma --max-model-len 8192` (run by Tensward in Docker with `--tool-call-parser gemma4`, which Tensward took from the checkpoint)
- change: `limit-mm-per-prompt` and `enable-auto-tool-choice`

This is the first of the 13 recorded answers to prompt `invoice-03` (the prompt that returned HTTP 400 in the baseline), re-serialised as indented JSON for reading. The other 12 are identical in the text field. Prompt text: `examples/prompts-images.jsonl`.

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

```json
{
  "finish_reason": "stop",
  "prompt_id": "invoice-03",
  "request_id": "20261001T161639Z-3fa8:000010",
  "text": "I cannot file this invoice because the image provided is a placeholder template with no visible text, supplier name, invoice number, date, or total amount.",
  "tool_calls": [],
  "truncated": false
}
```
