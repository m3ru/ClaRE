#!/bin/bash
#SBATCH -J rwr_claude_fixed
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Experiment 1 (PDF): retrain the claude_rwr rewriter on the CORRECTED-scorer
# Claude data (padding-fixed re-score = dataset_research_framing_full_shards_FIXED,
# staged to scratch/rwr_claude_data_fixed) with the ORIGINAL bin weighting
# (num_bins=5, bin_weights=[0,0,0,1,16] = train on top 40%, top quintile up 16x).
# The pre-existing rwr_claude_checkpoints was trained on the pre-fix (buggy) data,
# so this re-train is the apples-to-apples "original binning, bug-free scorer" run.
set -euo pipefail
echo "[stage] rwr_claude_fixed (Exp1) on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
python -c "import torch,transformers,peft,sentence_transformers" || { echo "[error] env broken"; exit 3; }
echo "[stage] python: $(python --version)"; nvidia-smi | head -12 || true

RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"
SHARD_DIR="${SCRATCH_BASE}/rwr_claude_data_fixed"
OUT_DIR="${SCRATCH_BASE}/rwr_claude_fixed_checkpoints"
[ -d "${SHARD_DIR}" ] || { echo "[error] missing shards ${SHARD_DIR}"; exit 2; }
mkdir -p "${OUT_DIR}"
cd "${RWR_DIR}"

python -u train_rwr.py \
  --shard_dir "${SHARD_DIR}" \
  --output_dir "${OUT_DIR}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1.5e-5 \
  --max_seq_length 512 \
  --num_bins 5 \
  --bin_weights 0,0,0,1,16 \
  --similarity_floor 0.5 \
  --seed 42 \
  --gradient_checkpointing

echo "[done] rwr_claude_fixed (Exp1) complete at $(date)"
ls -la "${OUT_DIR}"/final 2>&1 || true
