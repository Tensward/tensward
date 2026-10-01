# Changelog

## 0.1.3 (2026-10-01)

Mixture-of-experts and image+text models (validated with Gemma 4 26B-A4B AWQ on vLLM v0.30).

- Image+text and mixture-of-experts checkpoints register. `init` and `inspect` show the
  checkpoint's anatomy: components, attention layers, experts and input types.
- `init` and `inspect` estimate whether the model fits your GPU (`fit`) before anything starts.
  The estimate is advice, never a refusal, and memory used by other processes is reported
  separately.
- Hardware ceilings for mixture-of-experts models count the experts each decode step reads, and
  prefill uses the active parameters. Vision encoders and the embedding gather are no longer
  counted as decode traffic.
- `analyse` reports the engine's measured KV-cache capacity next to the estimate.
- New suggestion `no-media-encoders`: when no prompt sends images, serve only the text model
  (`--engine-arg language-model-only`). The batch-size suggestion no longer goes below what an
  image+text engine accepts.
- Gemma 4 tool calls use the engine's `gemma4` parser automatically. The MoE kernel backend
  appears in the report.
- `-tp` / `-pp` in `--current` are read as tensor and pipeline parallelism.
- Fixes:
  - A tensor that appears in two weight files is refused (its bytes were counted twice).
  - A precision mismatch between the configuration and the checkpoint now names what the
    checkpoint provides.

## 0.1.2 (2026-10-01)

- Fix: concurrent writes of the same state file (for example `serve stop` while `serve start`
  is still saving its state) could fail with "No such file or directory": every atomic write
  now uses its own temporary file.

## 0.1.1 (2026-10-01)

- `tensward --help` and the package description now say what the open-source package does:
  profile and diagnose (it does not tune automatically).
- Releases are published from GitHub Actions with PyPI trusted publishing.

## 0.1.0 (2026-10-01)

First public release: `init`, `inspect`, `analyse` (Docker or local runtime, `--trace`,
experimental `--counters`) and `serve`, with vLLM v0.30 as the first engine.
