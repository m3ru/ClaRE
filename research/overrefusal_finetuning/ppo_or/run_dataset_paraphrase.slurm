#!/bin/bash
#SBATCH -J or_gptoss
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 14:00:00
#SBATCH --output=or_gptoss_%A_%a.out
#SBATCH --error=or_gptoss_%A_%a.err
#SBATCH --array=0-31

set -euo pipefail
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- Array index → (dataset, shard_id, num_shards) ----
#   0-7:   cybermetric  (2000 prompts, 8 shards of ~250)
#   8-17:  sciq         (3000 prompts, 10 shards of ~300)
#  18-27:  medqa        (3000 prompts, 10 shards of ~300)
#  28-31:  pubmedqa     (1000 prompts, 4 shards of ~250)
IDX=${SLURM_ARRAY_TASK_ID}
if   (( IDX <= 7  )); then DATASET_NAME="cybermetric"; SHARD_ID=$((IDX));     NUM_SHARDS=8
elif (( IDX <= 17 )); then DATASET_NAME="sciq";        SHARD_ID=$((IDX-8));   NUM_SHARDS=10
elif (( IDX <= 27 )); then DATASET_NAME="medqa";       SHARD_ID=$((IDX-18));  NUM_SHARDS=10
elif (( IDX <= 31 )); then DATASET_NAME="pubmedqa";    SHARD_ID=$((IDX-28));  NUM_SHARDS=4
else echo "[error] Unknown array index ${IDX}"; exit 1
fi
echo "[job] Dataset: ${DATASET_NAME}, shard ${SHARD_ID}/${NUM_SHARDS} (array idx ${IDX})"

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

# ---- Conda ----
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ENVS_PATH}/or-para-env"
if [ ! -d "${ENV_PREFIX}" ]; then
  echo "[setup] Creating conda env: ${ENV_PREFIX}"
  conda create -y -p "${ENV_PREFIX}" python=3.11
  conda activate "${ENV_PREFIX}"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install transformers accelerate bitsandbytes sentence-transformers numpy tqdm datasets kernels "triton>=3.4.0"
else
  conda activate "${ENV_PREFIX}"
fi

# Ensure gpt-oss deps (kernels + triton) are present
GPTOSS_MARKER="${ENV_PREFIX}/.gptoss_deps_ok"
if [ ! -f "${GPTOSS_MARKER}" ]; then
  echo "[setup] Installing gpt-oss dependencies (kernels, triton)..."
  pip install -U kernels "triton>=3.4.0"
  touch "${GPTOSS_MARKER}"
else
  echo "[setup] gpt-oss deps already installed"
fi

echo "[env] Python: $(python --version)"
echo "[env] Node: $(hostname)"
echo "[env] GPUs:"; nvidia-smi || true

# ---- Paths ----
SCRIPT_DIR="${HOME}/ppo_or"
DATASET_DIR="${HOME}/scratch/ppo_or_datasets"
REFUSAL_VECTOR="${HOME}/scratch/ppo_or_data/refusal_vector.layer032.npz"

# ---- Step 1: Prepare dataset (only if not already done) ----
DATA_FILE="${DATASET_DIR}/${DATASET_NAME}.jsonl"
if [ ! -f "${DATA_FILE}" ]; then
  echo "[step1] Downloading and preparing ${DATASET_NAME}..."
  python -u "${SCRIPT_DIR}/prepare_datasets.py" --output_dir "${DATASET_DIR}" --dataset "${DATASET_NAME}"
else
  echo "[step1] ${DATA_FILE} already exists, skipping download"
fi

if [ ! -f "${DATA_FILE}" ]; then
  echo "[error] Data file not found after prep: ${DATA_FILE}" >&2
  exit 2
fi
echo "[step1] Dataset ready: $(wc -l < "${DATA_FILE}") prompts"

# ---- Step 2: Run paraphrase and score ----
OUTDIR="${HOME}/scratch/ppo_or_data/or_gptoss_${DATASET_NAME}"
mkdir -p "${OUTDIR}"
OUTPUT_FILE="${OUTDIR}/or_susceptibility_rankings.json"

echo "[step2] Running paraphrase_and_score on ${DATASET_NAME} (shard ${SHARD_ID}/${NUM_SHARDS})..."
echo "[step2] Gen model: openai/gpt-oss-120b (2x L40S, device_map=auto)"
python -u "${SCRIPT_DIR}/paraphrase_and_score.py" \
  --data_path "${DATA_FILE}" \
  --refusal_vector_path "${REFUSAL_VECTOR}" \
  --output "${OUTPUT_FILE}" \
  --gen_model "openai/gpt-oss-120b" \
  --scoring_model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --max_prompts 3000 \
  --n_paraphrases 20 \
  --temperature 0.9 \
  --activation_layer 32 \
  --score_batch_size 8 \
  --checkpoint_every 25 \
  --gen_batch_size 5 \
  --shard_id "${SHARD_ID}" \
  --num_shards "${NUM_SHARDS}"

echo "[done] ${DATASET_NAME} shard ${SHARD_ID}/${NUM_SHARDS} complete"
echo "[done] Output dir: ${OUTDIR}"
echo ""
echo "To merge all shards for ${DATASET_NAME}:"
echo "  python ${SCRIPT_DIR}/merge_shards.py --output_dir ${OUTDIR} --num_shards ${NUM_SHARDS}"
