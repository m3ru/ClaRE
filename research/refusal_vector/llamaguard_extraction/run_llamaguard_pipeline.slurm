#!/bin/bash
#SBATCH -J llamaguard_pipeline
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-l40s|gpu-a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Phase A of the Llama-Guard mirror pipeline:
#   1. Extract Llama-Guard-3-8B activations on the same benign + refusal prompt
#      set the Llama-3-8B vector was built on (final_benign_prompts.csv,
#      final_refusals_prompts.csv).
#   2. Compute diff-of-means refusal vector across all 32 layers, auto-select
#      the max-L2 layer.
#   3. Rescore dolly paraphrases.json with score_and_rank_all.py using
#      Llama-Guard-3-8B as the activation source and the new refusal vector.
#      Outputs land under paraphrases_api/dolly/llamaguard/ranked.json so the
#      original Llama-3-8B-scored ranked.json stays intact.

set -euo pipefail
echo "[stage] llamaguard pipeline start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- HF token ----
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

# ---- Caches on scratch (avoid 30G home quota and xet's home-dir reconstructor) ----
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
EXTRACT_PY="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/extract_activations_sharded.py"
COMPUTE_PY="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/compute_refusal_vector.py"
SCORE_PY="${PROJECT_ROOT}/research/overrefusal_finetuning/ppo_or/score_and_rank_all.py"
BENIGN_INPUT="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/final_benign_prompts.csv"
REFUSAL_INPUT="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/final_refusals_prompts.csv"
PARAPHRASES_ROOT="${PROJECT_ROOT}/research/overrefusal_finetuning/paraphrases_api"

LLAMAGUARD_DIR="${PROJECT_ROOT}/research/refusal_vector/llamaguard_extraction"
BENIGN_OUT_DIR="${SCRATCH_BASE}/llamaguard_acts_benign"
REFUSAL_OUT_DIR="${SCRATCH_BASE}/llamaguard_acts_refusal"
VECTOR_OUT="${LLAMAGUARD_DIR}/refusal_vector_llamaguard.npz"

MODEL="${MODEL:-meta-llama/Llama-Guard-3-8B}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LAST_K="${LAST_K:-1}"

for f in "${EXTRACT_PY}" "${COMPUTE_PY}" "${SCORE_PY}" "${BENIGN_INPUT}" "${REFUSAL_INPUT}"; do
  [ -f "$f" ] || { echo "[error] missing: $f"; exit 2; }
done
[ -d "${PARAPHRASES_ROOT}/dolly" ] || { echo "[error] missing paraphrases dolly dir"; exit 2; }

mkdir -p "${BENIGN_OUT_DIR}" "${REFUSAL_OUT_DIR}" "${LLAMAGUARD_DIR}"

echo "[config] model=${MODEL} batch=${BATCH_SIZE}"
echo "[config] benign_in=${BENIGN_INPUT} ($(wc -l < ${BENIGN_INPUT}) lines)"
echo "[config] refusal_in=${REFUSAL_INPUT} ($(wc -l < ${REFUSAL_INPUT}) lines)"
echo "[config] vector_out=${VECTOR_OUT}"
echo "[config] paraphrases_root=${PARAPHRASES_ROOT}"

# ---- Step 1: extract benign ----
echo "[step 1/4] Extracting activations for benign prompts with ${MODEL}..."
python -u "${EXTRACT_PY}" \
  --input "${BENIGN_INPUT}" \
  --output_base "${BENIGN_OUT_DIR}/acts" \
  --model "${MODEL}" \
  --num_shards 1 --shard_idx 0 \
  --batch_size "${BATCH_SIZE}" --dtype bf16 --last_k "${LAST_K}" \
  --layers all

# ---- Step 2: extract refusal ----
echo "[step 2/4] Extracting activations for refusal prompts with ${MODEL}..."
python -u "${EXTRACT_PY}" \
  --input "${REFUSAL_INPUT}" \
  --output_base "${REFUSAL_OUT_DIR}/acts" \
  --model "${MODEL}" \
  --num_shards 1 --shard_idx 0 \
  --batch_size "${BATCH_SIZE}" --dtype bf16 --last_k "${LAST_K}" \
  --layers all

# ---- Step 3: diff-of-means + max-L2 layer selection ----
echo "[step 3/4] Computing refusal vector (diff of means)..."
python -u "${COMPUTE_PY}" \
  --benign_dir "${BENIGN_OUT_DIR}" \
  --refusal_dir "${REFUSAL_OUT_DIR}" \
  --output "${VECTOR_OUT}" \
  --select_layer max

# Identify the selected single-layer file
SELECTED_VEC=$(ls "${LLAMAGUARD_DIR}"/refusal_vector_llamaguard.layer*.npz 2>/dev/null | head -1)
[ -n "${SELECTED_VEC}" ] || { echo "[error] compute_refusal_vector didn't produce a .layerNNN.npz"; exit 3; }
LAYER_NUM=$(basename "${SELECTED_VEC}" | sed 's/.*layer0*\([0-9]\+\).*/\1/')
echo "[step 3/4] selected vector: ${SELECTED_VEC}  (layer ${LAYER_NUM})"

# ---- Step 4: rescore dolly paraphrases with Llama-Guard + new vector ----
echo "[step 4/4] Rescoring dolly paraphrases with ${MODEL} + new vector..."
python -u "${SCORE_PY}" \
  --paraphrases_root "${PARAPHRASES_ROOT}" \
  --refusal_vector_path "${SELECTED_VEC}" \
  --scoring_model "${MODEL}" \
  --activation_layer "${LAYER_NUM}" \
  --datasets dolly \
  --output_subdir llamaguard

echo "[done] pipeline complete"
ls -la "${LLAMAGUARD_DIR}"/*.npz 2>&1 || true
ls -la "${PARAPHRASES_ROOT}/dolly/llamaguard/" 2>&1 || true
