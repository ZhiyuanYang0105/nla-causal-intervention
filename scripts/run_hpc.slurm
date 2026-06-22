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
# Faithful NLA at 1.5B on A100/H100. Three stages: build data -> warm-start + GRPO -> steg.
#
#   STEP 1 (login node, internet):   bash scripts/hpc_setup.sh     # env + models + FineWeb slice
#   STEP 2 (edit SBATCH lines + the conda line below for your cluster)
#   STEP 3:                          sbatch scripts/run_hpc.slurm
#
# After hpc_setup.sh this job runs FULLY OFFLINE (pre-downloaded models + local corpus), so it
# works on air-gapped compute nodes and wastes no GPU time on installs/downloads.
set -euo pipefail

# ---------------------------------------------------------------- knobs
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"          # run `sbatch` from the repo root
RUN=exp05_hpc
N=20000                                          # (h_l, summary) pairs to build
M=Qwen/Qwen2.5-1.5B                              # target model (activations + AV/AR init)
LAYER=14
POOLING=last                                     # paper's final-token readout (fallback: mean)
SUMMARIZER=Qwen/Qwen2.5-7B-Instruct              # strong rewriter/summarizer (real drift control)
WS_EPOCHS=4
GRPO_STEPS=1000
N_EVAL=500

# ---------------------------------------------------------------- environment
cd "$PROJECT_DIR"
module load cuda/12.4 2>/dev/null || true        # edit/remove for your cluster
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nla                               # created by hpc_setup.sh

export SCRATCH="${SCRATCH:-$PROJECT_DIR/.scratch}"
export HF_HOME="$SCRATCH/hf"                      # MUST match hpc_setup.sh
export TOKENIZERS_PARALLELISM=false; export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# fail fast with a clear message if the env wasn't set up
python -c "import torch,transformers,peft,sentence_transformers,nla_intervention" 2>/dev/null \
  || { echo "ENV NOT READY — run 'bash scripts/hpc_setup.sh' on a login node first"; exit 1; }
python -c "import torch;assert torch.cuda.is_available();print('CUDA:',torch.cuda.get_device_name(0))"

# offline corpus from hpc_setup.sh -> run with no network; else stream FineWeb (needs internet)
TEXT_FILE_ARG=""
if [ -f data/raw/fineweb_slice.jsonl ]; then
  TEXT_FILE_ARG="--text-file data/raw/fineweb_slice.jsonl"
  export HF_HUB_OFFLINE=1                         # models are pre-cached; no HF calls at runtime
  echo "offline mode: local corpus + cached models"
else
  echo "online mode: streaming FineWeb (needs internet on this node)"
fi

# ---------------------------------------------------------------- pipeline
echo "==== Stage 1/3: build data (harvest $N activations + summaries) ===="
python scripts/build_nla_data.py --out "$RUN" --n "$N" \
    --target-model "$M" --layer "$LAYER" --pooling "$POOLING" --summarizer "$SUMMARIZER" $TEXT_FILE_ARG

echo "==== Stage 2/3: warm-start + GRPO ===="
python scripts/train_nla.py --config "experiments/$RUN/config.yaml" \
    --grpo --ws-epochs "$WS_EPOCHS" --grpo-steps "$GRPO_STEPS"

echo "==== Stage 3/3: steganography intervention test ===="
python scripts/steg_intervention.py --config "experiments/$RUN/config.yaml" --n-eval "$N_EVAL"

echo "==== DONE. Results in results/$RUN/ (metrics.csv, intervention_*, ckpt/) ===="
