#!/bin/bash
#SBATCH -J rwr_v1_lg_eval
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Phase C: eval the Llama-Guard-trained RWR adapter.
# Generator = Llama-3-8B + adapter (what the LoRA sits on).
# Scorer    = Llama-Guard-3-8B + Llama-Guard refusal vector (the target).

set -euo pipefail
echo "[stage] v1-LG eval start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

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

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX="${SCRATCH_BASE}/conda_envs/ppo-or-env"
[ -d "${ENV_PREFIX}" ] || { echo "[error] env missing: ${ENV_PREFIX}"; exit 2; }
conda activate "${ENV_PREFIX}"
echo "[stage] python: $(python --version)"

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ClaRE}"
RWR_DIR="${PROJECT_ROOT}/research/overrefusal_finetuning/rwr_overrefusal"
ADAPTER_DIR="${ADAPTER_DIR:-${SCRATCH_BASE}/rwr_v1_llamaguard_dolly_checkpoints/final}"
SHARD_DIR="${SHARD_DIR:-${SCRATCH_BASE}/rwr_llamaguard_dolly_data}"
LLAMAGUARD_DIR="${PROJECT_ROOT}/research/refusal_vector/llamaguard_extraction"
OUTPUT_DIR="${OUTPUT_DIR:-${RWR_DIR}/eval_v1_llamaguard_dolly}"

[ -d "${ADAPTER_DIR}" ] || { echo "[error] missing adapter: ${ADAPTER_DIR}"; exit 2; }
[ -d "${SHARD_DIR}" ] || { echo "[error] missing shard dir: ${SHARD_DIR}"; exit 2; }

SELECTED_VEC=$(ls "${LLAMAGUARD_DIR}"/refusal_vector_llamaguard.layer*.npz 2>/dev/null | head -1)
[ -n "${SELECTED_VEC}" ] || { echo "[error] no llamaguard single-layer vector"; exit 2; }
echo "[config] using refusal vector: ${SELECTED_VEC}"

mkdir -p "${OUTPUT_DIR}"

cd "${RWR_DIR}"
python -u eval_v1_dolly.py \
  --adapter_dir "${ADAPTER_DIR}" \
  --shard_dir "${SHARD_DIR}" \
  --refusal_vector_path "${SELECTED_VEC}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --scoring_model meta-llama/Llama-Guard-3-8B \
  --output_dir "${OUTPUT_DIR}" \
  --n_per_prompt 5 \
  --temperature 0.7 \
  --max_eval_prompts 200 \
  --seed 42 \
  --sources dolly_val alpaca

echo "[done] eval complete"
ls -la "${OUTPUT_DIR}" || true
