#!/bin/bash
#SBATCH -J lg_vs_refusal
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Correlate Llama-Guard-3-8B P(unsafe) delta vs our Llama-3 refusal-circuit delta
# over the Claude Haiku rewrites (FIXED shards). ref_delta is read from the shards
# (not recomputed); only P(unsafe) is computed here.
set -euo pipefail
echo "[stage] lg_vs_refusal on $(hostname) at $(date)"
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
python -c "import torch,transformers,scipy" || { echo "[error] env broken"; exit 3; }

RWR_DIR="${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal"
cd "${RWR_DIR}"
SHARD_DIR="${SCRATCH_BASE}/rwr_claude_data_fixed"
[ -d "${SHARD_DIR}" ] || { echo "[error] missing shards ${SHARD_DIR}"; exit 2; }

python -u lg_vs_refusal_corr.py \
  --shard_dir "${SHARD_DIR}" \
  --out_dir prompt_iteration_results/lg_vs_refusal \
  --lg_model meta-llama/Llama-Guard-3-8B \
  --batch_size 24 \
  --sim_hi 0.85

echo "[done] lg_vs_refusal complete at $(date)"
