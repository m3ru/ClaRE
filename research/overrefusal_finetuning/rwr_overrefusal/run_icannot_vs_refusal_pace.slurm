#!/bin/bash
#SBATCH -J icannot_vs_refusal
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

# Correlate refusal-vector activation delta vs 'I cannot' logprob delta over the
# Claude rewrites, both on Llama-3-8B, same formatting, same model instance.
set -euo pipefail
echo "[stage] icannot_vs_refusal on $(hostname) at $(date)"
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

RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"
cd "${RWR_DIR}"
SHARD_DIR="${SCRATCH_BASE}/rwr_claude_data_fixed"
VEC="${HOME}/ClaRE/research/refusal_vector/Vector_Extraction/refusal_vector.layer032.npz"
[ -d "${SHARD_DIR}" ] || { echo "[error] missing shards ${SHARD_DIR}"; exit 2; }
[ -f "${VEC}" ] || { echo "[error] missing vector ${VEC}"; exit 2; }

python -u icannot_vs_refusal_corr.py \
  --shard_dir "${SHARD_DIR}" \
  --vector "${VEC}" \
  --out_dir prompt_iteration_results/icannot_vs_refusal \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --activation_layer 32 \
  --batch_size 24 \
  --sim_hi 0.85

echo "[done] icannot_vs_refusal complete at $(date)"
