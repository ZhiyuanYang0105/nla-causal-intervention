.PHONY: install dev test lint format nla-data train-nla steg clean

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
nla-data:                     # harvest activations + generate summaries
	python scripts/build_nla_data.py --out exp04_nla --n 2500

train-nla:                    # warm-start SFT + GRPO joint training (needs GPU)
	python scripts/train_nla.py --config experiments/exp04_nla/config.yaml --grpo --grpo-steps 200

steg:                         # steganography intervention test on the co-trained NLA
	python scripts/steg_intervention.py --config experiments/exp04_nla/config.yaml --n-eval 350

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
