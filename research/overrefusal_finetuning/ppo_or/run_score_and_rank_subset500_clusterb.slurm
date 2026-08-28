#!/bin/bash
#SBATCH -J or_score_subset500
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-l40s|gpu-a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Subset hedge: score + rank only the first 500 prompts per dataset.
# Writes outputs under {dataset}/subset500/ so it doesn't overwrite the
# full ranked.json that the companion full-scale job produces.

set -euo pipefail
echo "[stage] subset500 job start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- HF token ----
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

# ---- Scratch caches ----
export SCRATCH_BASE="${HOME}/scratch"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda_pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda_envs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${TMPDIR}"

export HF_HOME="${HOME}/scratch/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HOME}/scratch/torch_cache"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX="${CONDA_ENVS_PATH}/ppo-or-env"
[ -d "${ENV_PREFIX}" ] || { echo "[error] conda env missing: ${ENV_PREFIX}"; exit 2; }
conda activate "${ENV_PREFIX}"

# ---- Paths ----
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR}}"
SCORE_SCRIPT="${SCRIPT_DIR}/score_and_rank_all.py"
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/scratch/ClaRE}"
PARAPHRASES_ROOT="${PARAPHRASES_ROOT:-${PROJECT_ROOT}/research/overrefusal_finetuning/paraphrases_api}"
REFUSAL_VECTOR_PATH="${REFUSAL_VECTOR_PATH:-${PROJECT_ROOT}/research/refusal_vector/3_Vector_Extraction/refusal_vector.layer032.npz}"

SCORING_MODEL="${SCORING_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
ACTIVATION_LAYER="${ACTIVATION_LAYER:-32}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
MAX_PROMPTS_PER_DATASET="${MAX_PROMPTS_PER_DATASET:-500}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-subset500}"

echo "[paths] SCORE_SCRIPT=${SCORE_SCRIPT}"
echo "[paths] PARAPHRASES_ROOT=${PARAPHRASES_ROOT}"
echo "[paths] REFUSAL_VECTOR_PATH=${REFUSAL_VECTOR_PATH}"
echo "[config] MAX_PROMPTS_PER_DATASET=${MAX_PROMPTS_PER_DATASET}"
echo "[config] OUTPUT_SUBDIR=${OUTPUT_SUBDIR}"

[ -f "${SCORE_SCRIPT}" ] || { echo "[error] missing ${SCORE_SCRIPT}"; exit 2; }
[ -d "${PARAPHRASES_ROOT}" ] || { echo "[error] missing ${PARAPHRASES_ROOT}"; exit 2; }
[ -f "${REFUSAL_VECTOR_PATH}" ] || { echo "[error] missing ${REFUSAL_VECTOR_PATH}"; exit 2; }

python -u "${SCORE_SCRIPT}" \
  --paraphrases_root "${PARAPHRASES_ROOT}" \
  --refusal_vector_path "${REFUSAL_VECTOR_PATH}" \
  --scoring_model "${SCORING_MODEL}" \
  --activation_layer ${ACTIVATION_LAYER} \
  --batch_size ${SCORE_BATCH_SIZE} \
  --max_prompts_per_dataset ${MAX_PROMPTS_PER_DATASET} \
  --output_subdir "${OUTPUT_SUBDIR}"

echo "[done] subset500 ranked outputs under ${PARAPHRASES_ROOT}/*/${OUTPUT_SUBDIR}/"
