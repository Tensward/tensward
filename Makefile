.PHONY: sync check lint typecheck test dupcheck

sync:
	uv sync --locked

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

test:
	uv run pytest -q

# Fails on any clone of 6+ lines / 60+ tokens (Node is preinstalled on ubuntu-latest).
dupcheck:
	npx --yes jscpd@4.3.0 --min-lines 6 --min-tokens 60 --threshold 0 --format python \
		--ignore '**/tests/**' --reporters console src

check: lint typecheck test dupcheck
