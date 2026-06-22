#!/bin/bash
#SBATCH --job-name=nla-1p5b
#SBATCH --partition=gpu                 # <-- change to your cluster's GPU partition
#SBATCH --gres=gpu:1                    # 1 GPU; request A100-80GB or H100 (see --constraint)
##SBATCH --constraint=a100_80gb         # <-- uncomment/edit for your cluster's 80GB tag
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=nla-%j.out
#SBATCH --error=nla-%j.err
#
# Faithful NLA at 1.5B on A100/H100. Three stages: build data -> warm-start + GRPO -> steg test.
#   sbatch scripts/run_hpc.slurm
# Edit the SBATCH lines + the env block for your cluster. 80GB GPU recommended.
set -euo pipefail

# ---------------------------------------------------------------- knobs
PROJECT_DIR="$SLURM_SUBMIT_DIR"                 # run `sbatch` from the repo root
RUN=exp05_hpc
N=20000                                         # (h_l, summary) pairs to build
M=Qwen/Qwen2.5-1.5B                             # target model (activations + AV/AR init)
LAYER=14
POOLING=last                                    # paper's final-token readout (fallback: mean)
SUMMARIZER=Qwen/Qwen2.5-7B-Instruct             # strong rewriter/summarizer (real drift control)
WS_EPOCHS=4
GRPO_STEPS=1000
N_EVAL=500

# ---------------------------------------------------------------- environment
cd "$PROJECT_DIR"
# Option A: modules + conda (typical HPC). Edit to your site.
module load cuda/12.4 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nla 2>/dev/null || { conda create -y -n nla python=3.11 && conda activate nla; }

# Put HF/torch caches on fast scratch (avoid home-quota blowups). Edit SCRATCH.
export SCRATCH="${SCRATCH:-$PROJECT_DIR/.scratch}"
export HF_HOME="$SCRATCH/hf"; export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false; export PYTHONUNBUFFERED=1
# export HF_ENDPOINT=https://hf-mirror.com    # uncomment if HF is blocked on your cluster

python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q -e ".[models,nlp]" && pip install -q peft
python -c "import torch;print('CUDA:',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------- pipeline
echo "==== Stage 1/3: build data (harvest $N activations + 7B summaries) ===="
python scripts/build_nla_data.py --out "$RUN" --n "$N" \
    --target-model "$M" --layer "$LAYER" --pooling "$POOLING" --summarizer "$SUMMARIZER"

echo "==== Stage 2/3: warm-start + GRPO ===="
python scripts/train_nla.py --config "experiments/$RUN/config.yaml" \
    --grpo --ws-epochs "$WS_EPOCHS" --grpo-steps "$GRPO_STEPS"

echo "==== Stage 3/3: steganography intervention test ===="
python scripts/steg_intervention.py --config "experiments/$RUN/config.yaml" --n-eval "$N_EVAL"

echo "==== DONE. Results in results/$RUN/ (metrics.csv, intervention_*, ckpt/) ===="
