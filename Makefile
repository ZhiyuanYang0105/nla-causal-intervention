.PHONY: install dev test lint format pilot run analyze clean

install:
	pip install -e .

dev:
	pip install -e ".[models,nlp,dev]"

test:
	pytest -q

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

# End-to-end pipeline dry-run with fake AV/AR (no GPU/models). Writes results/dryrun/.
dryrun:
	python scripts/dry_run.py

# Validate the staged local-pilot orchestration with fakes (no models, runs anywhere).
local-pilot-smoke:
	python scripts/run_local_pilot.py --config experiments/exp01_pilot/config.yaml --smoke

# Real local pilot on MacBook (needs: pip install -e ".[models,mlx,nlp]").
local-pilot:
	python scripts/run_local_pilot.py --config experiments/exp01_pilot/config.yaml

# Legacy single-shot runner stub.
pilot:
	python scripts/run_experiment.py --config experiments/exp01_pilot/config.yaml

# Run a full experiment: make run CONFIG=experiments/exp01_pilot/config.yaml
run:
	python scripts/run_experiment.py --config $(CONFIG)

# Statistical analysis over a finished run: make analyze RUN=results/<run_id>
analyze:
	python scripts/analyze_results.py --run $(RUN)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
