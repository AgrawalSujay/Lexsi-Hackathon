PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: test lint format

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
