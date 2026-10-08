# Contributing

Thanks for helping. This page covers setup, how we test, and how to add an engine.

To add a command or a trace analysis as a separate package instead, see
[`docs/extending.md`](docs/extending.md).

## Setup

You need Python 3.12, [uv](https://docs.astral.sh/uv/) and Node.js (only for the duplicate check).
No GPU, model or network is needed to develop or test.

```sh
git clone https://github.com/Tensward/tensward && cd tensward
make sync     # uv sync --locked
make check    # ruff (lint and format check) + mypy + pytest + dupcheck
uv run tensward --help
```

`make check` must pass before a change is merged. The pieces can be run alone:
`make lint`, `make typecheck`, `make test`, `make dupcheck`.

## How we test

- **End to end first.** The main tests run the real `tensward` entry point (`main([...])`)
  against `tests/fake_vllm_server.py`, a stdlib-only stand-in for
  `vllm serve` that speaks the same HTTP API and metrics format. A new feature should be
  reachable from such a test.
- **Unit tests only for important logic**: parsers, arithmetic, anything with tricky edge
  cases. Do not add a unit test for every function.
- **Inputs are synthetic.** `tests/registration_fixtures.py` builds tiny checkpoints in a
  temporary directory. Nothing in the suite needs a real model.
- **A green fake suite does not prove the engine integration works.** Every real-engine
  surprise so far (renamed metrics, floats where integers were expected, checkpoint size
  limits) showed up only on a GPU. When you change an
  engine adapter, say in the pull request which real engine version you checked it against, and
  update the fake so it behaves like the real thing.

## No duplication

`make dupcheck` runs jscpd over the source and fails on any clone of 6 or more lines (60 tokens).
Extract the shared piece instead of copying it. Tests are excluded from the check.

## Adding an engine

An engine is one package in `src/tensward/engines/` that implements the
`Engine` protocol in `engines/protocol.py`, registered in `engines/__init__.py` (the `ENGINES`
dict, which also feeds `--engine`). `engines/vllm/` is the reference.

The protocol asks for:

- attributes: `name`, `label`, the `formats` and `platforms` it serves, `default_image`,
  `local_command`, `api_key_env`, the health, models and metrics paths, the server-log
  patterns, and `defaults` (what the engine does for each neutral setting left unset, which the
  report prints);
- `signals`: where the engine's Prometheus metrics export each `EngineSignals` field, as a map of
  `prometheus.Family` entries; a field the engine does not export is left out of the map (see
  `signal_source` below for what Tensward actually polls);
- `launch_argv(settings, ...)`: map the engine-neutral `Settings` (concurrency, context length,
  KV memory fraction, prefill batch tokens, prefix caching, ...) to the engine's command line;
- `parse_setup(text)`: understand a user's `serve`/`docker run` command as `Settings`;
- `recognizes`, `availability` and `inherited_env`: whether a command launches the engine,
  whether it is installed here, and which environment variables change how it behaves;
- the memory model: `default_kv_memory_fraction`, `memory_overhead_bytes`,
  `kv_in_flight_tokens`, `parallel_degree` and `max_graph_batch`;
- `with_engine_arg(settings, "KEY=VALUE")` and `engine_args_between(before, after)`: apply one
  `--engine-arg`, and say which ones turn one setup into another;
- `count_prompt_tokens` (the server's own token count of a prompt) and `request_extras` (fields
  the engine needs in each request body, such as structured output);
- `tool_calling_fix`, `reproducibility_advice` and `loads_encoders`: the engine's answers to
  "why can't these tools be served", "how can answers be made repeatable" and "are the media
  encoders loaded";
- `tracing`: the profiler behind `--trace` (`start`, `stop`, `collect`, the per-step annotation
  names `step_scope` and `step_pattern`, and its launch
  environment and arguments), or None when the engine has none, which `--trace` then refuses;
- `family`: the name that playbook and extension entries use in `Entry.engines` to say they suit
  this engine;
- `capabilities(version, settings)`: what the engine reports and can change, as a `Capabilities`:
  each `EngineSignals` field with its quality (exact, derived, approximate or absent), the levers
  it can realise, whether it supports traces and kernel counters, whether concurrent requests
  share one cached prefix copy and the smallest prefix its cache can reuse. It sets
  `resolves_at_launch` when it chooses values at launch that Tensward should read back, and
  `polls_metrics=False` when it has no metrics endpoint to poll;
- `lever_value` and `with_lever`: read and set the levers that are not plain `Settings` fields,
  in the engine's own flags;
- `signal_source`: what Tensward polls for signals. `read` does one poll, `from_run` adds what
  the finished run shows beyond the polls, and `replica_labels` tell data-parallel replicas
  apart. `metrics_path` and `signals` stay for compatibility; the signal source is
  authoritative;
- `probe` and `request_model`: a readiness probe for one wait (a `ReadyProbe` with `timeout_s` and
  `check(client, url, served_name)`, which returns `Ready`, `NotYet` or
  `Failed(reason)`), and the `model` value sent in the probe and in every request;
- `resolved` and `effective`: read what the engine chose at launch (from the server log Tensward
  passes in or from the engine's API), and fill those values into
  settings for diagnosis (the stored settings stay as requested);
- `default_profile`: the id of the calibration profile used when the platform, GPU or engine
  version is unknown, or None when the engine has no public profile;
- `engine_rules`, `predicates` and `hold_backs`: diagnosis rules only this engine has,
  playbook predicates and gates named `"<engine>:<name>"`, and reasons to hold an entry back on
  a run;
- `untimed`, `weights_on_device`, `runs_remote_code`, `frontend_processes` and `host_warnings`:
  why a request's client timings do not measure the engine, whether some weights stay in host
  memory, whether the settings let the engine run the checkpoint's own code, how many API-server
  processes a launch runs, and problems with this machine's installation;
- `log_prefix`, `quant_kernel_symbols`, `quantization_family`, `default_tool_parser`,
  `parse_quant_kernels` and the playbook (`playbook(version=...)`, entries loaded from
  `data/playbook.toml` for the levers the engine realises, and `consistent`, `unstartable`).

Everything engine-specific (flag names, metric names, image, environment) stays in that package;
the rest of the code never names an engine. Keep metric names next to a comment saying which
engine version they were checked against. Add a fake server for the new engine and an end-to-end
test like the existing ones.

The playbook's entries are data, in `tensward/data/playbook.toml` (in the package source); that
file is edited only in Tensward's private repository, so propose a new entry as an issue.

## Pull requests

Keep them small and focused, with a message that says why. Describe how you tested the change,
including any real-GPU run.

## Contributor License Agreement

Before your first pull request can be merged, you need to sign the
[Contributor License Agreement](CLA.md). The CLA Assistant bot comments on your pull request with a
link; signing takes one click with your GitHub account and covers all your future contributions.
You keep the copyright to your work; the agreement grants the project the rights it needs to
distribute it.
