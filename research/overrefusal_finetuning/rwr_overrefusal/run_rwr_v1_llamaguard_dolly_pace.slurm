#!/bin/bash
#SBATCH -J rwr_v1_lg_dolly
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Phase B of the Llama-Guard mirror pipeline:
# RWR training on the Llama-Guard-rescored dolly paraphrases.
# Uses v1 settings (k=18.4, c=0.5, weights=[1,2,4,8,16]) per user preference —
# v3 generations had similarity-to-original too low to be useful rewrites.

set -euo pipefail
echo "[stage] RWR v1-Llama-Guard-dolly start on $(hostname) at $(date)"
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
LG_RANKED="${PROJECT_ROOT}/research/overrefusal_finetuning/paraphrases_api/dolly/llamaguard/ranked.json"
SHARD_STAGE_DIR="${SCRATCH_BASE}/rwr_llamaguard_dolly_data"
OUTPUT_DIR="${SCRATCH_BASE}/rwr_v1_llamaguard_dolly_checkpoints"

[ -f "${LG_RANKED}" ] || { echo "[error] missing rescored ranked.json: ${LG_RANKED}"; exit 2; }
[ -d "${RWR_DIR}" ] || { echo "[error] missing RWR dir: ${RWR_DIR}"; exit 2; }

# Stage the rescored ranked.json under the filename pattern load_shards globs for
mkdir -p "${SHARD_STAGE_DIR}"
SHARD_FILE="${SHARD_STAGE_DIR}/or_susceptibility_rankings_shard0.json"
if [ ! -f "${SHARD_FILE}" ] || [ "${LG_RANKED}" -nt "${SHARD_FILE}" ]; then
  echo "[stage] copying ${LG_RANKED} -> ${SHARD_FILE}"
  cp -f "${LG_RANKED}" "${SHARD_FILE}"
fi
echo "[stage] shard file size: $(du -sh ${SHARD_FILE} | cut -f1)"

# v1 recipe (per user preference — v3 had similarity too low)
SIMILARITY_EXPONENT="${SIMILARITY_EXPONENT:-18.4}"
SIMILARITY_CENTER="${SIMILARITY_CENTER:-0.5}"
REFUSAL_DIVISOR="${REFUSAL_DIVISOR:-100.0}"
BIN_WEIGHTS="${BIN_WEIGHTS:-1,2,4,8,16}"
SIMILARITY_FLOOR="${SIMILARITY_FLOOR:-0.5}"
NUM_BINS="${NUM_BINS:-5}"

NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-1.5e-5}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
NUM_SAMPLES_PER_EPOCH="${NUM_SAMPLES_PER_EPOCH:-100000}"

mkdir -p "${OUTPUT_DIR}"

echo "[config] shard_dir=${SHARD_STAGE_DIR}"
echo "[config] output_dir=${OUTPUT_DIR}"
echo "[config] v1 recipe: k=${SIMILARITY_EXPONENT} c=${SIMILARITY_CENTER} d=${REFUSAL_DIVISOR} weights=${BIN_WEIGHTS}"
echo "[config] epochs=${NUM_EPOCHS} bs=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} samples/ep=${NUM_SAMPLES_PER_EPOCH}"

cd "${RWR_DIR}"
python -u train_rwr.py \
  --shard_dir "${SHARD_STAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LEARNING_RATE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --num_bins "${NUM_BINS}" \
  --similarity_floor "${SIMILARITY_FLOOR}" \
  --bin_weights "${BIN_WEIGHTS}" \
  --similarity_exponent "${SIMILARITY_EXPONENT}" \
  --similarity_center "${SIMILARITY_CENTER}" \
  --refusal_divisor "${REFUSAL_DIVISOR}" \
  --num_samples_per_epoch "${NUM_SAMPLES_PER_EPOCH}" \
  --seed 42 \
  --gradient_checkpointing

echo "[done] training complete; checkpoints under ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}" || true
