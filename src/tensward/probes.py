"""Short external commands for detection, bounded in time: a probe never hangs and never
raises."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence

PROBE_TIMEOUT_S = 10.0


def _probe(argv: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(argv), capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def run_probe(argv: Sequence[str]) -> str | None:
    """Standard output of ``argv``; None when it is missing, fails, prints undecodable text or
    runs longer than ``PROBE_TIMEOUT_S``."""
    done = _probe(argv)
    return done.stdout if done and done.returncode == 0 else None


@dataclass(frozen=True, slots=True)
class DockerImage:
    """What ``docker image inspect`` said. ``labels`` is None unless the image is present;
    otherwise ``reason`` is ``docker_missing`` (no docker CLI), ``docker_unreachable`` (it did
    not answer: the daemon is stopped, access is denied, or it timed out) or ``image_absent``,
    and ``detail`` is docker's first line of complaint."""

    labels: dict[str, str] | None
    reason: str | None = None
    detail: str = ""


def docker_image(image: str) -> DockerImage:
    """The labels of a locally present docker image (empty when it has none, or docker's answer
    cannot be read), or why they could not be had."""
    if shutil.which("docker") is None:
        return DockerImage(None, "docker_missing")
    done = _probe(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image])
    if done is None:
        return DockerImage(None, "docker_unreachable", "docker did not answer in time")
    if done.returncode != 0:
        lines = done.stderr.strip().splitlines()
        detail = lines[0] if lines else ""
        return DockerImage(
            None,
            "image_absent" if "no such image" in detail.lower() else "docker_unreachable",
            detail,
        )
    try:
        labels = json.loads(done.stdout)
    except ValueError:
        return DockerImage({})
    if not isinstance(labels, dict):
        return DockerImage({})
    return DockerImage({str(k): str(v) for k, v in labels.items()})


def executable(command: str) -> str | None:
    """The file the first word of ``command`` runs: a path as given, else the one found on
    PATH; None when there is none or it cannot be run."""
    try:
        first = shlex.split(command)[0]
    except (ValueError, IndexError):
        return None
    path = first if os.sep in first else shutil.which(first)
    return path if path and os.path.isfile(path) and os.access(path, os.X_OK) else None


def script_interpreter(path: str) -> str | None:
    """The interpreter named by the shebang line of the script at ``path``, when it is an
    absolute path; None for a binary, an ``env`` shebang or an unreadable file."""
    try:
        with open(path, "rb") as script:
            first = script.readline(512).decode("utf-8", "replace")
    except OSError:
        return None
    words = first[2:].split() if first.startswith("#!") else []
    return words[0] if words and os.path.isabs(words[0]) else None
