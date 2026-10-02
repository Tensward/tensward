# Changelog

## 0.2.0 (2026-10-02)

Foundations for more engines, checkpoint formats and hardware. Nothing you measure changes.

- `tensward env` shows what Tensward detects on this machine: the GPUs with their driver and
  CUDA version, whether each engine is available as a docker image or a local command (and its
  version), the checkpoint formats it registers, and the engine, format and GPU combinations
  that work here. `--json` prints one object, for agents. `--engine`, `--runtime`, `--image` and
  `--local-command` check a specific setup. It starts nothing and writes nothing.
- `init` records the engine. It is the one your `--current` command runs, else `--engine`, else
  the first engine that serves the checkpoint on this machine (vLLM, for a safetensors
  checkpoint on NVIDIA). `init` prints the choice and why, and the JSON of `init` and `inspect`
  gains `"environment": {"platform", "engine", "engine_choice", "format"}`. Projects registered
  earlier are vLLM projects and keep their identities.
- `analyse` and `serve` use the project's engine. `--engine` on them is now a check: naming
  another engine than the project's is refused (`project_config_unsupported`).
- `analyse` and `serve` now refuse before starting when the engine isn't available (exit 2,
  `engine_unavailable`); before, the same setups failed while launching. The message says what
  is wrong (Docker missing, the daemon unreachable, the image absent, or the local command
  missing) and how to fix it. `init` only warns.
- Every `report.md` starts with one line saying what ran: the engine and its version, the image
  or command, the GPU with its driver and CUDA version, and the checkpoint format and
  quantization. `run.json` records `engine`, `engine_version`, `platform` and `format`.
- A directory of GGUF files is refused with a message saying GGUF comes with the llama.cpp
  engine (coming in a later release).
- With no GPU detected, the hardware ceilings and `--require-gpu` say "no supported accelerator
  detected".

## 0.1.6 (2026-10-01)

Output quality in every before/after: see whether a faster setup changed the answers.

- `analyse --engine-arg ...` compares the run with the newest run of your current setup: a
  verdict on the answers judged against that setup's own noise, the speed change, reference
  match and format health, three answer pairs in `report.md`, and a private
  `compare/<baseline>-vs-<candidate>/answers.md` with every prompt's answers side by side. The
  terminal prints the verdict. `--no-retain-responses` turns it off.
- `tensward compare --project P BASELINE_RUN CANDIDATE_RUN` does the same for any two runs of a
  project.
- Every `analyse` run writes `run.json`, recording what produced it. Runs from earlier versions
  cannot be compared.
- The suggestion that may change the outputs says that the report then compares the answers.
- `--require-equal` (on `analyse` and `compare`) exits with status 4 unless every answer is one
  the baseline gave for that prompt: the same text, tool calls and finish reason, over every
  successful answer. The report gives the first differing character per prompt and how often the
  current setup reproduced its own answers, with the batch-invariant mode as advice when it did
  not. `--baseline-answers FILE` compares with recorded production answers instead of a run.
- Every `analyse` run records each GPU's name, driver, PCI device id, VBIOS and SM count, and
  the newest CUDA the driver supports, in `run.json`. A comparison warns when the GPU or the
  driver differs. `--require-gpu NAME` and `--require-driver VERSION` refuse a mismatched machine
  before the model loads (`gpu_mismatch`).
- On machines with mixed GPU models, `--runtime local` now numbers GPUs as nvidia-smi does
  (`CUDA_DEVICE_ORDER=PCI_BUS_ID`, unless you export it yourself).
- Fix: a suggested command replaces an override you already gave, instead of repeating the flag.

## 0.1.5 (2026-10-01)

Fixes from a first run on a real production command and checkpoint.

- Serving flags that count tokens, such as `--long-prefill-token-threshold 0`, are kept in the
  imported command instead of being treated as secrets. `HF_TOKEN`, `--hf-token` and `--api-key`
  are still blanked. `-cc` is read as `--compilation-config`, and dotted compilation keys keep
  their underscores. Re-running `init` with a command that has such a flag, `-cc` or a dotted
  compilation key now gives `project_inputs_changed`: register the project again.
- Suggestions that raise concurrency now raise a pinned `cudagraph_capture_sizes` (or its
  maximum) with it, so the larger batches keep their CUDA graphs. Other suggestions leave pinned
  sizes alone, and a report check says when running sequences exceeded what the graphs cover.
  The capture sizes are read from `--compilation-config` and `--max-cudagraph-capture-size`;
  `--cudagraph-capture-sizes` and the dotted capture-size keys are refused at import with the
  spelling to use.
- The n-gram suggestion turns async scheduling off, which vLLM v0.30 requires, and says so. It
  is listed as not applicable, with the reason, when pinned CUDA graph sizes cannot hold its
  steps, or when it would shrink the batch the graphs cover. A suggestion that the engine's
  rules reduce to the current setup is dropped.
- `--quantization auto_gptq` (and the other names vLLM resolves to the checkpoint's method) is
  accepted on a matching checkpoint; a different method is still refused.
- GPTQModel checkpoints register: `quant_log.csv` is ignored and the paths under
  `quantization_config.meta` are not treated as file references.
- The configuration accepts `open_loop` and `capped` arrivals, as the documentation says. Fit
  checks one sequence, or the declared cap, when the workload sets no concurrency.
- Fit leaves out the vision and audio weights when the text model alone is served
  (`--language-model-only`, or every media limit at 0).
- Not changed: the 4-characters-per-token estimate, which is wrong for non-Latin scripts.
  `stop`, `structured_output` and `logprobs` still cannot be set in the configuration.

## 0.1.4 (2026-10-01)

Image workloads (validated with Gemma 4 26B-A4B AWQ on vLLM v0.30, NVIDIA A10G).

- Chat prompts can carry images as local files: OpenAI `image_url` parts with a path relative to
  the prompts file. The rules:
  - formats are PNG, JPEG and WebP, checked from their headers;
  - animated images are refused;
  - the bounds are 20 MiB and 40 megapixels per image, 16 images per prompt and 1 GiB per
    workload;
  - remote and inline (`data:`) URLs are refused, with a message that says how to fix them.
- Images are hashed into the workload identity. `inspect` and `analyse` name an image that
  changed, and an image that changed after registration is never sent.
- Images are sent as `data:` URLs, built when each request is sent through a bounded cache.
  Measurements count image tokens and split requests with and without images.
- Fit counts images at their maximum token count. It also counts the larger batch the engine
  uses while media inputs are on: on an A10G it is within 1% of the measured KV capacity.
- New setting `media_limits` (`--limit-mm-per-prompt`). `no-media-encoders` is not suggested
  when prompts send images.
- A request whose image changed after registration is reported as such, not as "no HTTP
  response".
- New example: `examples/prompts-images.jsonl` with generated invoice, table, chart and dashboard
  images (`examples/images/make_images.py`), and `examples/config-images.json`.

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
