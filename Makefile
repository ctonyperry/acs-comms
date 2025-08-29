# Makefile for ACS Bridge development tasks

.PHONY: help install install-dev install-all fmt lint type test run clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

install-all:  ## Install all dependencies (prod + dev + optional)
	pip install -e ".[dev,stt,tts,llm]"

fmt:  ## Format code with black
	black src/ tests/ run.py

lint:  ## Lint code with ruff
	ruff check src/ tests/ run.py

type:  ## Type check with mypy
	mypy src/

test:  ## Run tests
	pytest tests/

run:  ## Run the application locally
	python run.py

run-uvicorn:  ## Run with uvicorn directly
	uvicorn src.acs_bridge.main:app --reload --host 0.0.0.0 --port 8080

clean:  ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

setup-hooks:  ## Setup pre-commit hooks
	pre-commit install

all-checks:  ## Run all checks (format, lint, type, test)
	$(MAKE) fmt
	$(MAKE) lint
	$(MAKE) type
	$(MAKE) test