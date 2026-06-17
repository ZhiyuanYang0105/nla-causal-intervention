.PHONY: install dev test lint format dryrun local-pilot-smoke local-pilot analyze \
        nla-data train-nla steg clean

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

# ---- Phase 2: FAITHFUL NLA (activation-injection AV + co-trained AR) ----
# Build (h_l, summary) data, then warm-start + GRPO, then the steganography test.
nla-data:
	python scripts/build_nla_data.py --reuse exp02_open --out exp04_nla --n 2500

train-nla:
	python scripts/train_nla.py --config experiments/exp04_nla/config.yaml --grpo --grpo-steps 200

steg:
	python scripts/steg_intervention.py --config experiments/exp04_nla/config.yaml --n-eval 350

# ---- Phase 1: text-proxy pilot (CONCLUSION RETRACTED — see docs/nla_faithful_findings.md) ----
# Kept for the record; its AV/AR were not co-trained so it cannot test steganography.
dryrun:                       # fake AV/AR plumbing demo, no models
	python scripts/dry_run.py
local-pilot-smoke:            # staged orchestration with fakes (no models)
	python scripts/run_local_pilot.py --config experiments/exp01_pilot/config.yaml --smoke
local-pilot:                  # real text-proxy pilot
	python scripts/run_local_pilot.py --config experiments/exp01_pilot/config.yaml
analyze:                      # paired stats over a finished phase-1 run: make analyze RUN=results/<id>
	python scripts/analyze_results.py --run $(RUN)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
