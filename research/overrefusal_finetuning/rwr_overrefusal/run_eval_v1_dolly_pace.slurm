#!/bin/bash
#SBATCH -J rwr_v1_dolly_eval
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
echo "[stage] v1-dolly eval start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- HF token ----
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

# ---- Caches on scratch ----
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_XET_CACHE="${HF_HOME}/xet"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${SCRATCH_BASE}/torch_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
export HF_HUB_DISABLE_XET=1
export HF_XET_DISABLE=1
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${HF_XET_CACHE}" "${TORCH_HOME}" "${TMPDIR}"

# ---- Conda env ----
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX="${SCRATCH_BASE}/conda_envs/ppo-or-env"
[ -d "${ENV_PREFIX}" ] || { echo "[error] env missing: ${ENV_PREFIX}"; exit 2; }
conda activate "${ENV_PREFIX}"
echo "[stage] python: $(python --version)"
nvidia-smi | head -20 || true

# ---- Paths ----
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ClaRE}"
RWR_DIR="${PROJECT_ROOT}/research/overrefusal_finetuning/rwr_overrefusal"
ADAPTER_DIR="${ADAPTER_DIR:-${SCRATCH_BASE}/rwr_v1_dolly_checkpoints/final}"
SHARD_DIR="${SHARD_DIR:-${SCRATCH_BASE}/rwr_dolly_v1_data}"
REFUSAL_VECTOR_PATH="${REFUSAL_VECTOR_PATH:-${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/refusal_vector.layer032.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${RWR_DIR}/eval_v1_dolly}"

[ -d "${ADAPTER_DIR}" ] || { echo "[error] missing adapter: ${ADAPTER_DIR}"; exit 2; }
[ -d "${SHARD_DIR}" ] || { echo "[error] missing shard dir: ${SHARD_DIR}"; exit 2; }
[ -f "${REFUSAL_VECTOR_PATH}" ] || { echo "[error] missing refusal vector: ${REFUSAL_VECTOR_PATH}"; exit 2; }

mkdir -p "${OUTPUT_DIR}"

N_PER_PROMPT="${N_PER_PROMPT:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_EVAL_PROMPTS="${MAX_EVAL_PROMPTS:-200}"

echo "[config] adapter_dir=${ADAPTER_DIR}"
echo "[config] shard_dir=${SHARD_DIR}"
echo "[config] refusal_vector=${REFUSAL_VECTOR_PATH}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] n_per_prompt=${N_PER_PROMPT} temperature=${TEMPERATURE} max_prompts=${MAX_EVAL_PROMPTS}"

cd "${RWR_DIR}"
python -u eval_v1_dolly.py \
  --adapter_dir "${ADAPTER_DIR}" \
  --shard_dir "${SHARD_DIR}" \
  --refusal_vector_path "${REFUSAL_VECTOR_PATH}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir "${OUTPUT_DIR}" \
  --n_per_prompt "${N_PER_PROMPT}" \
  --temperature "${TEMPERATURE}" \
  --max_eval_prompts "${MAX_EVAL_PROMPTS}" \
  --seed 42 \
  --sources dolly_val alpaca

echo "[done] eval complete; results under ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}" || true
