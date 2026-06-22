.PHONY: install dev test lint format smoke nla-data train-nla steg clean

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

# ---- Faithful NLA pipeline: build data -> warm-start + GRPO -> steganography test ----
# Tiny end-to-end smoke (0.5B, N=40) — validates the WHOLE pipeline runs before a paid GPU run.
smoke:
	python scripts/build_nla_data.py --out smoke --n 40 --target-model Qwen/Qwen2.5-0.5B \
	    --layer 12 --pooling last --summarizer Qwen/Qwen2.5-0.5B-Instruct
	python scripts/train_nla.py --config experiments/smoke/config.yaml --grpo --ws-epochs 1 --grpo-steps 2
	python scripts/steg_intervention.py --config experiments/smoke/config.yaml --n-eval 8

nla-data:                     # harvest activations + generate summaries
	python scripts/build_nla_data.py --out exp04_nla --n 2500

train-nla:                    # warm-start SFT + GRPO joint training (needs GPU)
	python scripts/train_nla.py --config experiments/exp04_nla/config.yaml --grpo --grpo-steps 200

steg:                         # steganography intervention test on the co-trained NLA
	python scripts/steg_intervention.py --config experiments/exp04_nla/config.yaml --n-eval 350

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
