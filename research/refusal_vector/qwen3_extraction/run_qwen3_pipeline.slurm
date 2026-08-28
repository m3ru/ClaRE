#!/bin/bash
#SBATCH -J qwen3_refusal_vec
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-l40s|gpu-a100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
echo "[stage] qwen3 refusal-vector pipeline start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- HF token ----
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

# ---- Caches on scratch (home quota is 30G — keep EVERYTHING on scratch) ----
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_XET_CACHE="${HF_HOME}/xet"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TORCH_HOME="${SCRATCH_BASE}/torch_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
# Prior attempt (job 5035769) hit "Disk quota exceeded" via HF xet downloader.
# Disable xet to force the legacy download path which we know writes to HF_HOME.
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
PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/anon-repo}"
QWEN_DIR="${PROJECT_ROOT}/research/refusal_vector/qwen3_extraction"
EXTRACT_PY="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/extract_activations_sharded.py"
COMPUTE_PY="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/compute_refusal_vector.py"
BENIGN_INPUT="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/final_benign_prompts.csv"
REFUSAL_INPUT="${PROJECT_ROOT}/research/refusal_vector/Vector_Extraction/final_refusals_prompts.csv"
BRIEF_MD="${QWEN_DIR}/v1_brief.md"
RESCORE_PY="${QWEN_DIR}/rescore_v1_brief.py"

BENIGN_OUT_DIR="${SCRATCH_BASE}/qwen3_acts_benign"
REFUSAL_OUT_DIR="${SCRATCH_BASE}/qwen3_acts_refusal"
mkdir -p "${BENIGN_OUT_DIR}" "${REFUSAL_OUT_DIR}"
VECTOR_OUT="${QWEN_DIR}/refusal_vector_qwen3.npz"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LAST_K="${LAST_K:-1}"

for f in "${EXTRACT_PY}" "${COMPUTE_PY}" "${BENIGN_INPUT}" "${REFUSAL_INPUT}" "${BRIEF_MD}" "${RESCORE_PY}"; do
  [ -f "$f" ] || { echo "[error] missing: $f"; exit 2; }
done

echo "[config] model=${MODEL} batch=${BATCH_SIZE} last_k=${LAST_K}"
echo "[config] benign_in=${BENIGN_INPUT}  n=$(wc -l < ${BENIGN_INPUT})"
echo "[config] refusal_in=${REFUSAL_INPUT} n=$(wc -l < ${REFUSAL_INPUT})"

# ---- Extract benign ----
echo "[step 1/4] Extracting activations for benign prompts..."
python -u "${EXTRACT_PY}" \
  --input "${BENIGN_INPUT}" \
  --output_base "${BENIGN_OUT_DIR}/acts" \
  --model "${MODEL}" \
  --num_shards 1 --shard_idx 0 \
  --batch_size "${BATCH_SIZE}" --dtype bf16 --last_k "${LAST_K}" \
  --layers all

# ---- Extract refusal ----
echo "[step 2/4] Extracting activations for refusal prompts..."
python -u "${EXTRACT_PY}" \
  --input "${REFUSAL_INPUT}" \
  --output_base "${REFUSAL_OUT_DIR}/acts" \
  --model "${MODEL}" \
  --num_shards 1 --shard_idx 0 \
  --batch_size "${BATCH_SIZE}" --dtype bf16 --last_k "${LAST_K}" \
  --layers all

# ---- Compute diff-of-means ----
echo "[step 3/4] Computing refusal vector (diff of means)..."
python -u "${COMPUTE_PY}" \
  --benign_dir "${BENIGN_OUT_DIR}" \
  --refusal_dir "${REFUSAL_OUT_DIR}" \
  --output "${VECTOR_OUT}" \
  --select_layer max

# ---- Rescore v1_brief pairs ----
echo "[step 4/4] Rescoring v1_brief adaptations..."
python -u "${RESCORE_PY}" \
  --brief "${BRIEF_MD}" \
  --vector "${VECTOR_OUT}" \
  --model "${MODEL}" \
  --select_layer max \
  --output "${QWEN_DIR}/rescore_v1_brief_qwen.json"

echo "[done] pipeline complete"
ls -la "${QWEN_DIR}"/*.npz "${QWEN_DIR}"/rescore_v1_brief_qwen.* 2>&1 || true
