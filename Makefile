# Makefile for Egg – a minimal RL incubator for language models.
#
# Usage:
#   make install    – create venv and install the package
#   make test       – run the test suite
#   make run        – run a quick baseline experiment
#   make clean      – remove build artifacts and the venv

PYTHON   ?= python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
PYTHON_V := $(VENV)/bin/python

.PHONY: install test run clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

install: $(VENV)/bin/activate  ## Create venv and install egg in editable mode
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[test]"
	@echo "\n✅  Installed.  Activate with:  source $(VENV)/bin/activate"

test: install  ## Run the test suite
	@# Each experiment registers global absl flags, so we isolate them.
	@for t in $$(find experiments -name '*_test.py'); do \
		echo "\n▶ $$t" && $(PYTEST) "$$t" -v || exit 1; \
	done
	@echo "\n✅  All tests passed."

run: install  ## Run a quick baseline experiment (10 steps)
	$(PYTHON_V) -m experiments.baseline.run --sweep.num_steps=10

clean:  ## Remove venv and build artifacts
	rm -rf $(VENV) build dist *.egg-info egg.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
