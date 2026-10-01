# Changelog

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
