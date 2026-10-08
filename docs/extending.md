# Extending Tensward

A separate package can add subcommands to `tensward` and its own analysis of a GPU trace. It
imports only `tensward.api`. Everything outside `tensward.api` is internal and may change in any
release.

## Entry points

An extension registers itself in its `pyproject.toml`:

```toml
[project.entry-points."tensward.commands"]
mytool = "my_package.cli:register"

[project.entry-points."tensward.analysis"]
mytool = "my_package.analysis:EXTENDER"
```

- `tensward.commands`: a callable `register(subcommands)` that adds parsers to the `tensward`
  command line (`subcommands` is the `argparse` subparsers object). Each parser sets
  `run=<function(arguments) -> int>` with `set_defaults`; the return value is the exit code.
- `tensward.analysis`: an `Extender`. Tensward uses the first one that loads.

Set `TENSWARD_DISABLE_PLUGINS` (to anything) to load no extension.

## Version check

`tensward.api.API_VERSION` is the extension API version, separate from the package version.
This document describes version 1.

When an extension loads, Tensward reads `Extender.api_version`, or an `API_VERSION` attribute on
a command's `register`. On a mismatch the extension is skipped with one line on standard error
and the command goes on:

- `<package> needs extension API 2; this Tensward has 1; upgrade tensward`
- `<package> was built for extension API 0; this Tensward has 1; upgrade <package>`

An analysis extension without a version is skipped with
`<package> does not declare an extension API version; upgrade <package>`. A command `register`
without an `API_VERSION` attribute is accepted. An extension that fails to import, to register
its commands or to analyse a trace is skipped with one line naming the error; it never stops
the command.

## Rules within version 1

- Names in `tensward.api.__all__` are not removed or renamed.
- The types an extension constructs are keyword-only: `Entry`, `Extension`, `Extender`,
  `Settings`, `Situation`, `Suggestion`, `WorkloadFacts` and the progress events.
- The fields of the other exported types are only appended, never inserted or reordered.
  Construct them with keywords too.
- Functions keep their positional parameters; new parameters are keyword-only with a default.
- Documented behaviour does not change. New names may be added in any release.
- A breaking change raises `API_VERSION`. The previous version stays loadable for one minor
  release.

Added within version 1 in 0.3.7, all with defaults that keep earlier callers working:

- `classify(measurement, settings, answers, *, engine=None, profile=None, facts=None,
  resolved=None, version=None, platform=None)`. With `engine=None` the run is diagnosed as vLLM,
  and with `profile=None` at the engine's default calibration profile. A caller diagnosing
  another engine passes `engine`, and the profile `profile_for(engine, platform, gpu, version)`
  returns. `platform` is the platform's name, such as `"nvidia"`; None means unknown.
- `applicable(..., *, engine=None)`: with `engine`, an extension's entries are kept only where
  the engine declares every lever the entry changes (`levers`) and allows each value the entry
  sets (`sets`, lever name to value), and only for this engine's family when
  `engines` names families (an entry with empty `engines` is not filtered by family). The
  engine's hold-backs apply to them.
- `Entry.levers`, `Entry.engines` and `Entry.sets`; `Measurement.engine_signals`,
  `Measurement.replicas` and `Measurement.untimed_requests`; `Trace.steps` and
  `read_trace(..., *, step_pattern=None)`; `Extender.profiles`.
- New `Engine` members (only Tensward implements engines; extensions may call them): `family`,
  `capabilities`, `lever_value`, `with_lever`, `default_profile`, `signal_source`, `probe`,
  `request_model`, `resolved`, `effective`, `weights_on_device`, `untimed`, `runs_remote_code`,
  `frontend_processes`, `hold_backs`, `engine_rules`, `host_warnings`, `predicates` and
  `playbook(*, version=None)`. `metrics_path` and `signals` stay for compatibility; the engine's
  `signal_source` is what Tensward polls.
- New names: `CalibrationProfile`, `Capabilities`, `SignalSupport`, `LeverSupport`,
  `Speculation`, `Resolved`, `LEVERS` and `profile_for`.
- `Capabilities.resolves_at_launch`: an engine that chooses settings at launch sets it. Tensward
  then passes the server log, once the server is ready, to the engine's `resolved`, which may
  read the log or the engine's API. `Capabilities.polls_metrics` is False for an engine with no
  metrics endpoint to poll.

Later releases may append fields in the same way (for example lever bounds on `LeverSupport`).

The fields of the exported types are covered, and these nested fields:

- `ResolvedProject`: `prompts`, `artifact`, `record.current_setup`,
  `config.workload.request_count`, `config.workload.arrival.kind` and
  `config.workload.arrival.concurrency`. Change a workload with `with_workload`.
- `Measurement.trace`: `status` and `analysis`.
- `Measurement.ceilings`: `unavailable`, `prefill_pct_of_ceiling` and `decode_pct_of_ceiling`.

Other nested fields may change between releases.

## Names

| Group | Names |
|---|---|
| Version | `API_VERSION` |
| Commands | `add_project_argument`, `add_runtime_arguments`, `add_slo_arguments`, `engine_for`, `runtime_for`, `slo_for`, `run_guarded` |
| Projects | `load_project`, `ResolvedProject`, `CurrentSetup`, `settings_for`, `with_workload`, `workload_facts`, `PromptEntry`, `CHARS_PER_TOKEN` |
| Measuring | `measure`, `Measurement`, `nearest_rank`, `describe_run`, `Subject`, `Slo`, `DEFAULT_SLO`, `DEFAULT_READY_TIMEOUT_S`, `Runtime`, `Engine`, `Settings`, `AnalyseFailure` |
| Diagnosis and playbook | `classify`, `Entry`, `Situation`, `Suggestion`, `WorkloadFacts`, `applicable`, `rungs` |
| Engines and calibration | `Capabilities`, `SignalSupport`, `LeverSupport`, `Resolved`, `LEVERS`, `Speculation`, `CalibrationProfile`, `profile_for` |
| Analysis extensions | `Extender`, `Extension`, `Trace`, `Event`, `Gap`, `ATTENTION`, `GEMM`, `GAP_MIN_US`, `CountersSummary`, `KernelCounters`, `short_name` |
| Serve source | `PACKAGES_FILE`, `PackagesFile`, `Package` |
| Files and errors | `new_run_id`, `write_json`, `write_jsonl`, `PreflightError`, `EXIT_OK`, `PROJECT_INPUTS_INVALID` |
| Progress | `emit`, `sink`, `Sink`, `Phase`, `ProgressEvent`, `PhaseStarted`, `ServerLoading`, `ServerReady`, `RequestsDone`, `WindowOpened`, `Note`, `RunWritten` |

## Contracts

### `measure`

`measure(project, *, engine, runtime, settings, run_dirs, retain_responses, ready_timeout_s,
environment, subject=Subject(), slo=DEFAULT_SLO, run_projects=None, steady_state=True,
engine_args=())` starts one launch, runs the workload into each of `run_dirs` in turn, stops the
server and returns one `Measurement` per run.

- `environment` is `describe_run(...)`, read once before the launch.
- `run_projects` gives each run its own variant of `project`, for example one made with
  `with_workload`. Every variant must have the same `artifact` and `record.current_setup` as
  `project`, otherwise `ValueError`.
- A closed-loop run offers a start-up wave first and measures after it. With
  `steady_state=False` it measures the whole run.
- Each run directory is complete when `measure` returns: the evidence, `metrics.json` with the
  diagnosis, `report.json`, `report.md` and `run.json`.

### `with_workload`

`with_workload(project, *, requests=None, concurrency=None)` returns the project offering exactly
`requests` (one each, in order) and/or keeping `concurrency` requests in flight. The changed
workload is validated as a registered one is: no requests, records of the other API, or a
concurrency on an arrival that does not take one raise `ValueError`.

### `rungs`

`rungs(entry, situation, *, prepare=...)` returns at most four settings, from the easiest step to
the full change. Each differs from `situation.settings` only in `entry.steps_on` (and the full
change's other settings). `prepare` adjusts each rung and returns None for one that cannot be
tried; that rung is dropped, and so is one that `prepare` leaves unchanged.

### `applicable`

`applicable(entries, situation, *, allow_quality_changes, near=False, engine=None)` returns `(found, gated)`:
the entries the situation calls for as `Suggestion`s, and the entries a gate rules out, each with
the gate's reason. The installed `Extender`'s entries join `entries`.

With `engine=`, an `Extender` entry is kept only where that engine declares every lever in the
entry's `levers` and allows each value in its `sets`, and for that engine's family when
`engines` names families; the engine's hold-backs then apply to it. An entry with no `levers`
is not filtered by capabilities.

### `Extender` and `Extension`

`Extender(api_version, analyse, entries=(), counters=None, profiles=())`, keyword-only:

- `analyse(trace, measurement, settings)` runs after `analyse --trace` and returns an
  `Extension(sections, result=None)`. `sections` are markdown lines, placed in the report's
  `extension` section. `result`, a dataclass, is stored in `metrics.json` as `trace.analysis`.
- `counters(summary, measurement)` runs after `analyse --counters` and returns an `Extension`;
  its `result` is stored as `counters.analysis`.
- `entries` are playbook entries that join the engine's in `applicable`. They can read
  `situation.measurement.trace.analysis`.
- `profiles` are calibration profiles. `profile_for` returns the engine's default profile when the
  platform, GPU or engine version is unknown, for a run with the weights on the device and not on
  a CPU. Otherwise it takes an extension profile whose `engine` equals the engine's `name`, whose
  `platform` matches and whose `devices` name the serving GPU, and falls back to the public
  profile for the platform and placement. Past the default, a profile is taken only where its
  `versions` holds for the engine version: `"*"` holds for every version, otherwise it is a
  PEP 440 specifier set such as `">=0.30,<0.31"`.

### `packages.json`

`tensward serve start --from <id>` serves a package from `<project>/optimize/<id>/packages.json`
(`PACKAGES_FILE`); `--from latest` takes the newest. Any extension may write one, with
`PackagesFile(...).model_dump(mode="json")`:

```json
{
  "schema_version": "1",
  "recommended": "balanced",
  "packages": [
    {
      "name": "balanced",
      "settings": {"max_concurrent_requests": 32, "prefix_caching": true},
      "requires_review": false,
      "confirmed": true
    }
  ]
}
```

- `recommended`: the package `serve` picks without `--package`, or null.
- `settings`: a `Settings` as JSON.
- `requires_review`: its changes can affect output quality.
- `confirmed`: it beat the current setup again when measured a second time.

Unknown fields are refused. A `packages.json` without `schema_version`, as the optimizer wrote it
before this format, is still read.

### Progress events

Long commands send `ProgressEvent`s to the current sink. The default prints the terminal's
progress lines on standard error. Inside `with sink(target):` events go to `target` instead,
including events from threads started with `asyncio.to_thread`. An extension reports its own
progress with `emit(Note(text=...))` or `emit(PhaseStarted(phase=..., detail=...))`.

| Event | Fields (besides `at_s`, seconds since the command began) |
|---|---|
| `PhaseStarted` | `phase`, `detail` (the terminal line; empty prints nothing) |
| `ServerLoading` | `phase`, `waited_s` (0 when loading begins, then every 15 s) |
| `ServerReady` | `phase`, `load_s` |
| `RequestsDone` | `phase`, `done`, `total` (sent for every completion) |
| `WindowOpened` | `lead` (requests in the start-up wave) |
| `Note` | `text` |
| `RunWritten` | `run_dir` |

`Phase` is one of `preflight`, `launch`, `warmup`, `measure`, `trace`, `counters`, `compare`,
`diagnose`, `write` and `stop`. Failures are exceptions, not events.

### `report.json`

Each run directory has `report.json`, the report as data; `report.md` and the terminal output
are rendered from it. Its format is described in [`cli.md`](cli.md#run-directory).

## Examples

A command extension:

```python
import argparse

from tensward.api import API_VERSION, EXIT_OK, add_project_argument, load_project, run_guarded


def _count(arguments: argparse.Namespace) -> int:
    def run() -> int:
        project = load_project(arguments.project)
        print(f"{len(project.prompts)} prompts")
        return EXIT_OK

    return run_guarded("count-prompts", run)


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("count-prompts", help="count the registered prompts")
    add_project_argument(parser, verify_weights=False)
    parser.set_defaults(run=_count)


register.API_VERSION = API_VERSION  # type: ignore[attr-defined]
```

An analysis extension:

```python
from dataclasses import dataclass

from tensward.api import API_VERSION, Extender, Extension, Measurement, Settings, Trace


@dataclass(frozen=True)
class KernelShare:
    kernels: int
    busy_pct: float


def analyse(trace: Trace, measurement: Measurement, settings: Settings) -> Extension:
    share = KernelShare(len(trace.kernels), 100 * trace.busy_time / trace.window)
    line = f"- {share.kernels} kernels kept the GPU busy {share.busy_pct:.0f}% of the trace"
    return Extension(sections=[line], result=share)


EXTENDER = Extender(api_version=API_VERSION, analyse=analyse)
```
