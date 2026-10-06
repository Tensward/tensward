"""Tensward's extension API: the only names an extension may import. ``__all__`` is the
contract; ``docs/extending.md`` states the rules. Every name lives in its home module; this
module only re-exports them.

API v1 covers the fields of the exported types and these nested fields:

- ``ResolvedProject``: ``prompts``, ``artifact``, ``record.current_setup``,
  ``config.workload.request_count``, ``config.workload.arrival.kind`` and
  ``config.workload.arrival.concurrency``. Change a workload with ``with_workload``.
- ``Measurement.trace``: ``status`` and ``analysis``.
- ``Measurement.ceilings``: ``unavailable``, ``prefill_pct_of_ceiling`` and
  ``decode_pct_of_ceiling``.

Other nested fields may change between releases.

The types an extension constructs are keyword-only: ``Entry``, ``Extension``, ``Extender``,
``Settings``, ``Situation``, ``Suggestion``, ``WorkloadFacts`` and the progress events. The
fields of the other exported types are only appended, never inserted or reordered; construct
them with keywords too. Functions keep their positional parameters; new ones are keyword-only.
"""

from .classify import classify
from .cli_options import (
    add_project_argument,
    add_runtime_arguments,
    add_slo_arguments,
    engine_for,
    run_guarded,
    runtime_for,
    slo_for,
)
from .engines.protocol import Engine
from .environment import describe_run
from .errors import (
    EXIT_OK,
    PROJECT_INPUTS_INVALID,
    AnalyseFailure,
    PreflightError,
)
from .extensions import API_VERSION, Extender, Extension
from .files import new_run_id, write_json, write_jsonl
from .fit import CHARS_PER_TOKEN
from .inputs import PromptEntry
from .measure import measure
from .measurement import Measurement, nearest_rank
from .packages_file import PACKAGES_FILE, Package, PackagesFile
from .playbook import Entry, Situation, Suggestion, WorkloadFacts, rungs
from .profiling.counters import CountersSummary, KernelCounters, short_name
from .profiling.trace import ATTENTION, GAP_MIN_US, GEMM, Event, Gap, Trace
from .progress import (
    Note,
    Phase,
    PhaseStarted,
    ProgressEvent,
    RequestsDone,
    RunWritten,
    ServerLoading,
    ServerReady,
    Sink,
    WindowOpened,
    emit,
    sink,
)
from .project import (
    CurrentSetup,
    ResolvedProject,
    load_project,
    settings_for,
    with_workload,
    workload_facts,
)
from .report.build import Subject
from .runtime import DEFAULT_READY_TIMEOUT_S, Runtime
from .settings import Settings
from .slo import DEFAULT_SLO, Slo
from .suggest import applicable

__all__ = [
    "API_VERSION", "ATTENTION", "AnalyseFailure", "CHARS_PER_TOKEN", "CountersSummary",
    "CurrentSetup", "DEFAULT_READY_TIMEOUT_S", "DEFAULT_SLO", "EXIT_OK", "Engine", "Entry",
    "Event", "Extender", "Extension", "GAP_MIN_US", "GEMM", "Gap", "KernelCounters",
    "Measurement", "Note", "PACKAGES_FILE", "PROJECT_INPUTS_INVALID", "Package", "PackagesFile",
    "Phase", "PhaseStarted", "PreflightError", "ProgressEvent", "PromptEntry", "RequestsDone",
    "ResolvedProject", "RunWritten", "Runtime", "ServerLoading", "ServerReady", "Settings",
    "Sink", "Situation", "Slo", "Subject", "Suggestion", "Trace", "WindowOpened",
    "WorkloadFacts", "add_project_argument", "add_runtime_arguments", "add_slo_arguments",
    "applicable", "classify", "describe_run", "emit", "engine_for", "load_project", "measure",
    "nearest_rank", "new_run_id", "run_guarded", "rungs", "runtime_for", "settings_for",
    "short_name", "sink", "slo_for", "with_workload", "workload_facts", "write_json",
    "write_jsonl",
]  # fmt: skip
