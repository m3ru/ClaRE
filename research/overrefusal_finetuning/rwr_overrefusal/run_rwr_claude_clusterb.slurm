#!/bin/bash
#SBATCH -J rwr_claude
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Reproduce the claude_rwr attacker on <cluster>: RWR-train Llama-3-8B-Instruct on the
# deduped Claude Haiku rewrites. Same recipe as run_rwr_claude_only.slurm
# (BinningConfig defaults: k=5.0, c=0.75, d=100, bin_weights=[0,0,0,1,16] = top-40%).
set -euo pipefail
echo "[stage] rwr_claude retrain on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export HF_DATASETS_CACHE="${HOME}/.cache/hf_datasets_relabel" TMPDIR="${HOME}/tmp_relabel"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
echo "[stage] python: $(python --version)"; nvidia-smi | head -12 || true

RWR_DIR="${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal"
SHARD_DIR="${SCRATCH_BASE}/rwr_claude_data"
OUT_DIR="${SCRATCH_BASE}/rwr_claude_checkpoints"
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
  --similarity_floor 0.5 \
  --seed 42 \
  --gradient_checkpointing

echo "[done] rwr_claude retrain complete at $(date)"
ls -la "${OUT_DIR}"/final 2>&1 || true
