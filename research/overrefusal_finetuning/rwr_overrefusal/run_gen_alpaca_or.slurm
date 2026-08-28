#!/bin/bash
#SBATCH -J gen_alpaca_or
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH --array=0-15%8
#SBATCH --output=slurm-genalpaca-%A_%a.out
#SBATCH --error=slurm-genalpaca-%A_%a.err

# Generate 4 OR adaptations per UNUSED alpaca prompt with the claude_rwr attacker.
# 16 shards x ~3064 prompts -> ~196k rewrites total. Each shard < 2h.
set -euo pipefail
echo "[stage] gen shard ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
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
echo "[stage] python: $(python --version)"

RWR_DIR="${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal"
ADAPTER="${ADAPTER:-${SCRATCH_BASE}/rwr_claude_checkpoints/final}"
OUT_DIR="${RWR_DIR}/gen_alpaca_or"
[ -d "${ADAPTER}" ] || { echo "[error] missing adapter ${ADAPTER}"; exit 2; }
mkdir -p "${OUT_DIR}"
cd "${RWR_DIR}"

python -u generate_or_alpaca.py \
  --adapter_dir "${ADAPTER}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir "${OUT_DIR}" \
  --num_shards 16 --shard_idx "${SLURM_ARRAY_TASK_ID}" \
  --n_per_prompt 4 --temperature 1.0 --top_p 0.9 \
  --max_new_tokens 256 --batch_size 8

echo "[done] gen shard ${SLURM_ARRAY_TASK_ID} complete at $(date)"
