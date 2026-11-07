.PHONY: test lint format

test:
	python -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
