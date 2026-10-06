"""Refusals: a fixed code, a message that never contains a prompt, a declared value or a secret
(it may name a file or a path the user gave), and the process exit status the CLI maps the
code to."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

EXIT_OK = 0
EXIT_REFUSED = 2  # the environment refuses: unsafe state, a busy project, an unexpected failure
EXIT_INVALID_INPUT = 3  # a declared document or checkpoint does not meet its contract
EXIT_ANSWERS_DIFFER = 4  # the run succeeded, but --require-equal could not show equal answers

# Refusal codes. The checkpoint_* codes come from registering a checkpoint directory, the
# project_* codes from the project record and its declared inputs.
CHECKPOINT_CHANGED = "checkpoint_changed"
CHECKPOINT_INVENTORY_UNEXPECTED = "checkpoint_inventory_unexpected"
CHECKPOINT_INVENTORY_UNSAFE = "checkpoint_inventory_unsafe"
CHECKPOINT_LAYOUT_INVALID = "checkpoint_layout_invalid"
CHECKPOINT_PRECISION_UNSUPPORTED = "checkpoint_precision_unsupported"
CHECKPOINT_UNSUPPORTED = "checkpoint_unsupported"
ENGINE_UNAVAILABLE = "engine_unavailable"
GPU_MISMATCH = "gpu_mismatch"
PROJECT_BUSY = "project_busy"
PROJECT_CONFIG_UNSUPPORTED = "project_config_unsupported"
PROJECT_INPUTS_CHANGED = "project_inputs_changed"
PROJECT_INPUTS_INVALID = "project_inputs_invalid"
PROJECT_LAYOUT_INVALID = "project_layout_invalid"
PROJECT_NOT_REGISTERED = "project_not_registered"
PROJECT_PUBLICATION_FAILED = "project_publication_failed"
PROJECT_RECORD_INVALID = "project_record_invalid"
PROJECT_SNAPSHOT_MISMATCH = "project_snapshot_mismatch"
PROJECT_STATE_UNSAFE = "project_state_unsafe"
RUNNER_FAILURE = "runner_failure"

# Refusals about the local environment exit 2; every other code is about a declared input.
ENVIRONMENT_CODES = frozenset(
    {
        CHECKPOINT_CHANGED,
        ENGINE_UNAVAILABLE,
        GPU_MISMATCH,
        PROJECT_BUSY,
        PROJECT_NOT_REGISTERED,
        PROJECT_PUBLICATION_FAILED,
        PROJECT_STATE_UNSAFE,
        RUNNER_FAILURE,
    }
)

MAX_VALIDATION_DETAILS = 3
# The type of a validation error whose message is already a complete sentence for the user.
USER_MESSAGE_ERROR = "user_message"


class AnalyseFailure(Exception):
    """The run could not be completed; the message is safe to print."""


class PreflightError(Exception):
    """A refusal to register or run: ``code`` is stable, ``message`` says what to fix."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def not_found_message(what: str, path: Path) -> str:
    return f"{what} not found at {path} (if running in a container, mount it at the same path)"


def exit_code_for(code: str) -> int:
    return EXIT_REFUSED if code in ENVIRONMENT_CODES else EXIT_INVALID_INPUT


def validation_summary(error: ValidationError) -> str:
    """Where a document was rejected and why, without echoing the submitted values. An error
    that carries its own sentence stands alone: the others are the branches of a union that
    rejected the same value."""
    for item in error.errors():
        if item["type"] == USER_MESSAGE_ERROR:
            return item["msg"]
    details = []
    for item in error.errors()[:MAX_VALIDATION_DETAILS]:
        location = ".".join(str(part) for part in item["loc"]) or "document"
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)
