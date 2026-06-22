#!/bin/bash
# Run ONCE on a LOGIN/DATA-TRANSFER node (needs internet; no GPU needed).
# Creates the env, installs deps, pre-downloads all models, and pre-fetches a FineWeb slice
# to a local file — so the GPU job (run_hpc.slurm) can run FULLY OFFLINE (many clusters
# air-gap compute nodes, and this also avoids burning paid GPU time on downloads/installs).
#
#   bash scripts/hpc_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."                          # repo root

# ---- knobs (keep in sync with experiments/exp05_hpc/config.yaml + run_hpc.slurm) ----
N="${N:-20000}"                                  # pairs the run will build
M="${M:-Qwen/Qwen2.5-1.5B}"                      # target model M (= AV/AR init)
SUMMARIZER="${SUMMARIZER:-Qwen/Qwen2.5-7B-Instruct}"
EMBEDDER="${EMBEDDER:-sentence-transformers/all-mpnet-base-v2}"
export SCRATCH="${SCRATCH:-$PWD/.scratch}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"         # MUST match run_hpc.slurm's HF_HOME
# export HF_ENDPOINT=https://hf-mirror.com       # uncomment if HF is blocked

echo "== 1. env =="
# conda (edit if your cluster uses venv/modules instead):
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nla 2>/dev/null || { conda create -y -n nla python=3.11 && conda activate nla; }
# torch with CUDA (edit cu124 to your cluster's CUDA), then the project + model/nlp extras:
python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[models,nlp]"

echo "== 2. pre-download models into $HF_HOME =="
# `huggingface-cli download` is a dead stub in huggingface-hub>=1.x (which transformers 5.x
# forces) — use `hf download`, falling back to the version-stable python API.
for repo in "$M" "$SUMMARIZER" "$EMBEDDER"; do
  echo "  downloading $repo ..."
  hf download "$repo" >/dev/null 2>&1 \
    || huggingface-cli download "$repo" >/dev/null 2>&1 \
    || python -c "from huggingface_hub import snapshot_download; snapshot_download('$repo')"
done

echo "== 3. pre-fetch FineWeb slice (offline corpus for the GPU job) =="
mkdir -p data/raw
PREFETCH=$((N * 2)) python - <<'PY'
import json, os
from datasets import load_dataset
n = int(os.environ["PREFETCH"]); out = "data/raw/fineweb_slice.jsonl"
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
with open(out, "w") as f:
    k = 0
    for r in ds:
        if k >= n: break
        if r.get("text"): f.write(json.dumps({"text": r["text"]}) + "\n"); k += 1
print(f"wrote {k} docs -> {out}")
PY

echo "== 4. verify =="
python -c "import torch,transformers,peft,datasets,sentence_transformers,nla_intervention; \
print('imports OK | transformers', transformers.__version__)"
echo "== DONE. Now: sbatch scripts/run_hpc.slurm (it will run offline from this cache) =="
