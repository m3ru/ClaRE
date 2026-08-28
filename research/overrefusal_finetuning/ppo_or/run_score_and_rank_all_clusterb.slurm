#!/bin/bash
#SBATCH -J or_score_rank_all
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-l40s|gpu-a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
echo "[stage] job start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- Hugging Face token ----
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
  echo "[auth] Loaded HF token from cache"
fi

# ---- Scratch-based caches ----
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

# ---- Conda environment (reuse ppo-or-env) ----
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ENVS_PATH}/ppo-or-env"
if [ ! -d "${ENV_PREFIX}" ]; then
  echo "[error] Conda env not found: ${ENV_PREFIX}" >&2
  echo "Run the training SLURM job first (or create env) before running this." >&2
  exit 2
fi
conda activate "${ENV_PREFIX}"
echo "[stage] conda env: ${ENV_PREFIX}"
echo "[stage] python: $(python --version)"

# ---- Dep check ----
python -u - <<'PY'
import importlib.util
need = ["torch", "transformers", "accelerate", "bitsandbytes",
        "sentence_transformers", "numpy"]
missing = [m for m in need if not importlib.util.find_spec(m)]
if missing:
    raise SystemExit(f"[deps] missing: {', '.join(missing)}")
print("[deps] ok")
PY

echo "Node: $(hostname)"
echo "GPU:"; nvidia-smi || true

# ---- Paths (override via env vars if needed) ----
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR}}"
SCORE_SCRIPT="${SCRIPT_DIR}/score_and_rank_all.py"

# Default: assume you staged the project under ~/scratch/anon-repo
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/scratch/anon-repo}"
PARAPHRASES_ROOT="${PARAPHRASES_ROOT:-${PROJECT_ROOT}/research/overrefusal_finetuning/paraphrases_api}"
REFUSAL_VECTOR_PATH="${REFUSAL_VECTOR_PATH:-${PROJECT_ROOT}/research/refusal_vector/3_Vector_Extraction/refusal_vector.layer032.npz}"

SCORING_MODEL="${SCORING_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
ACTIVATION_LAYER="${ACTIVATION_LAYER:-32}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
DATASETS_ARG="${DATASETS_ARG:-}"   # e.g. "cybermetric sciq"; empty = all
MAX_PROMPTS_PER_DATASET="${MAX_PROMPTS_PER_DATASET:-0}"   # 0 = all
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"   # e.g. "subset500"; empty = write to {ds}/

echo "[paths] SCORE_SCRIPT=${SCORE_SCRIPT}"
echo "[paths] PARAPHRASES_ROOT=${PARAPHRASES_ROOT}"
echo "[paths] REFUSAL_VECTOR_PATH=${REFUSAL_VECTOR_PATH}"
echo "[config] SCORING_MODEL=${SCORING_MODEL}"
echo "[config] ACTIVATION_LAYER=${ACTIVATION_LAYER}"
echo "[config] SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE}"
echo "[config] DATASETS=${DATASETS_ARG:-<all>}"
echo "[config] MAX_PROMPTS_PER_DATASET=${MAX_PROMPTS_PER_DATASET}"
echo "[config] OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-<none>}"

[ -f "${SCORE_SCRIPT}" ] || { echo "[error] missing ${SCORE_SCRIPT}"; exit 2; }
[ -d "${PARAPHRASES_ROOT}" ] || { echo "[error] missing ${PARAPHRASES_ROOT}"; exit 2; }
[ -f "${REFUSAL_VECTOR_PATH}" ] || { echo "[error] missing ${REFUSAL_VECTOR_PATH}"; exit 2; }

# ---- Launch ----
CMD="python -u \"${SCORE_SCRIPT}\" \
  --paraphrases_root \"${PARAPHRASES_ROOT}\" \
  --refusal_vector_path \"${REFUSAL_VECTOR_PATH}\" \
  --scoring_model \"${SCORING_MODEL}\" \
  --activation_layer ${ACTIVATION_LAYER} \
  --batch_size ${SCORE_BATCH_SIZE}"

if [ -n "${DATASETS_ARG}" ]; then
  CMD="${CMD} --datasets ${DATASETS_ARG}"
fi
if [ "${MAX_PROMPTS_PER_DATASET}" -gt 0 ]; then
  CMD="${CMD} --max_prompts_per_dataset ${MAX_PROMPTS_PER_DATASET}"
fi
if [ -n "${OUTPUT_SUBDIR}" ]; then
  CMD="${CMD} --output_subdir \"${OUTPUT_SUBDIR}\""
fi

echo "[launch] ${CMD}"
eval "${CMD}"

echo "[done] All datasets scored + ranked under ${PARAPHRASES_ROOT}/*/"
