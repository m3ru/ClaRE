#!/bin/bash
#SBATCH -J or_para_%a
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

SHARD_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=${NUM_SHARDS:-10}

echo "[stage] job start — shard ${SHARD_ID}/${NUM_SHARDS}"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- Hugging Face token ----
# Paste your HF token here, or run `huggingface-cli login` on an interactive node
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
  echo "[auth] Loaded HF token from cache"
fi
if [ -n "${HUGGING_FACE_HUB_TOKEN}" ]; then
  echo "[auth] HF_TOKEN set (starts with: ${HUGGING_FACE_HUB_TOKEN:0:10}...)"
else
  echo "[auth] WARNING: No HF token found! Gated models will fail."
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

# ---- Conda environment ----
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ENVS_PATH}/ppo-or-env"
if [ ! -d "${ENV_PREFIX}" ]; then
  echo "[error] Conda env not found: ${ENV_PREFIX}" >&2
  exit 2
fi
conda activate "${ENV_PREFIX}"

echo "[stage] conda env ready: ${ENV_PREFIX}"
echo "[stage] python: $(python --version)"

# ---- Extra deps for gpt-oss ----
GPTOSS_MARKER="${ENV_PREFIX}/.gptoss_deps_v3_ok"
if [ ! -f "${GPTOSS_MARKER}" ]; then
  echo "[stage] Installing gpt-oss dependencies..."
  pip install -U transformers kernels tqdm "triton>=3.4.0"
  touch "${GPTOSS_MARKER}"
else
  echo "[stage] gpt-oss deps already installed"
fi

python -u - <<'PY'
import importlib
need = ["torch", "transformers", "kernels", "accelerate", "bitsandbytes",
        "sentence_transformers", "numpy"]
missing = [m for m in need if not importlib.util.find_spec(m)]
if missing:
    raise SystemExit(f"[deps] missing: {', '.join(missing)}")
print("[deps] ok")
PY

echo "[env] Python: $(python --version)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:"; nvidia-smi || true

# ---- Paths ----
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR}}"
SCORE_SCRIPT="${SCRIPT_DIR}/paraphrase_and_score.py"

SCRATCH_DATA="${SCRATCH_DATA:-${HOME}/scratch/ppo_or_data}"
DATA_PATH="${DATA_PATH:-${SCRATCH_DATA}/databricks-dolly-15k.jsonl}"
REFUSAL_VECTOR_PATH="${REFUSAL_VECTOR_PATH:-${SCRATCH_DATA}/refusal_vector.layer032.npz}"

# ---- Configuration ----
GEN_MODEL="${GEN_MODEL:-openai/gpt-oss-120b}"
SCORING_MODEL="${SCORING_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
MAX_PROMPTS="${MAX_PROMPTS:-500}"
N_PARAPHRASES="${N_PARAPHRASES:-20}"
TEMPERATURE="${TEMPERATURE:-0.9}"
ACTIVATION_LAYER="${ACTIVATION_LAYER:-32}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-5}"

# Shared output directory across all shards (use array job ID, not task ID)
RUN_NAME="or_paraphrase_array_${SLURM_ARRAY_JOB_ID}"
OUTDIR="${HOME}/scratch/ppo_or_data/${RUN_NAME}"
mkdir -p "${OUTDIR}"
OUTPUT_FILE="${OUTDIR}/or_susceptibility_rankings.json"

echo "[paths] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[paths] SCORE_SCRIPT=${SCORE_SCRIPT}"
echo "[paths] DATA_PATH=${DATA_PATH}"
echo "[config] GEN_MODEL=${GEN_MODEL}"
echo "[config] SCORING_MODEL=${SCORING_MODEL}"
echo "[config] MAX_PROMPTS=${MAX_PROMPTS}"
echo "[config] SHARD=${SHARD_ID}/${NUM_SHARDS}"
echo "[config] OUTPUT_DIR=${OUTDIR}"

if [ ! -f "${SCORE_SCRIPT}" ]; then
  echo "[error] Script not found: ${SCORE_SCRIPT}" >&2
  exit 2
fi
if [ ! -f "${DATA_PATH}" ]; then
  echo "[error] Data file not found: ${DATA_PATH}" >&2
  exit 2
fi
if [ ! -f "${REFUSAL_VECTOR_PATH}" ]; then
  echo "[error] Refusal vector not found: ${REFUSAL_VECTOR_PATH}" >&2
  exit 2
fi

# ---- Launch shard ----
CMD="python -u \"${SCORE_SCRIPT}\" \
  --data_path \"${DATA_PATH}\" \
  --refusal_vector_path \"${REFUSAL_VECTOR_PATH}\" \
  --output \"${OUTPUT_FILE}\" \
  --gen_model \"${GEN_MODEL}\" \
  --scoring_model \"${SCORING_MODEL}\" \
  --max_prompts ${MAX_PROMPTS} \
  --n_paraphrases ${N_PARAPHRASES} \
  --temperature ${TEMPERATURE} \
  --activation_layer ${ACTIVATION_LAYER} \
  --score_batch_size ${SCORE_BATCH_SIZE} \
  --checkpoint_every ${CHECKPOINT_EVERY} \
  --gen_batch_size ${GEN_BATCH_SIZE} \
  --shard_id ${SHARD_ID} \
  --num_shards ${NUM_SHARDS}"

echo "[launch] ${CMD}"
eval "${CMD}"

echo "[done] Shard ${SHARD_ID} complete"
echo "[done] Output dir: ${OUTDIR}"

# If this is the last shard to finish, offer merge hint
echo ""
echo "To merge all shards:"
echo "  python ${SCRIPT_DIR}/merge_shards.py --output_dir ${OUTDIR} --num_shards ${NUM_SHARDS}"
