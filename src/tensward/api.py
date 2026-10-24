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

from .calibration import CalibrationProfile, profile_for
from .capabilities import Capabilities, LeverSupport, Resolved, SignalSupport
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
from .extensions import API_VERSION, Extender, Extension, Knowledge
from .facts import FACTS_VERSION, Facts, canonical_json
from .files import new_run_id, write_json, write_jsonl
from .fit import CHARS_PER_TOKEN
from .inputs import PromptEntry
from .kb_condition import ANCHOR_KEYS, Node, from_refs, terms_from_json
from .kb_local import kb_home
from .kb_schema import SCHEMA as KB_SCHEMA
from .kb_schema import Record, RecordHead, record_from_json, trusted_head, trusted_record
from .levers import LEVERS, Speculation
from .matcher import KnowledgeSource, Match, Score, Scorer
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
    "ANCHOR_KEYS", "API_VERSION", "ATTENTION", "AnalyseFailure", "CHARS_PER_TOKEN",
    "CalibrationProfile", "Capabilities", "CountersSummary", "CurrentSetup",
    "DEFAULT_READY_TIMEOUT_S", "DEFAULT_SLO", "EXIT_OK", "Engine", "Entry", "Event", "Extender",
    "Extension", "FACTS_VERSION", "Facts", "GAP_MIN_US", "GEMM", "Gap", "KB_SCHEMA",
    "KernelCounters", "Knowledge", "KnowledgeSource", "LEVERS", "LeverSupport", "Match",
    "Measurement", "Node", "Note", "PACKAGES_FILE", "PROJECT_INPUTS_INVALID", "Package",
    "PackagesFile", "Phase", "PhaseStarted", "PreflightError", "ProgressEvent", "PromptEntry",
    "Record", "RecordHead", "RequestsDone", "Resolved", "ResolvedProject", "RunWritten", "Runtime",
    "Score", "Scorer", "ServerLoading", "ServerReady", "Settings", "SignalSupport", "Sink",
    "Situation", "Slo", "Speculation", "Subject", "Suggestion", "Trace", "WindowOpened",
    "WorkloadFacts", "add_project_argument", "add_runtime_arguments", "add_slo_arguments",
    "applicable", "canonical_json", "classify", "describe_run", "emit", "engine_for", "from_refs",
    "kb_home", "load_project", "measure", "nearest_rank", "new_run_id", "profile_for",
    "record_from_json", "run_guarded", "rungs", "runtime_for", "settings_for",
    "short_name", "sink", "slo_for", "terms_from_json", "trusted_head", "trusted_record",
    "with_workload", "workload_facts", "write_json", "write_jsonl",
]  # fmt: skip
