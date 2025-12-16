#!/bin/bash
#SBATCH -J ppo_refusal
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "[stage] job start"

# If anything fails, print the line number (helps debug fast)
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- Hugging Face token (PASTE YOUR TOKEN TO THE RIGHT OF =) ----
# IMPORTANT: Do NOT hardcode tokens in this script.
export HUGGING_FACE_HUB_TOKEN=

# ---- Put conda/pip caches + envs in SCRATCH (avoid HOME quota) ----
export SCRATCH_BASE="${HOME}/scratch"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda_pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda_envs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${TMPDIR}"

# ---- Caches (PACE scratch recommended) ----
export HF_HOME="${HOME}/scratch/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HOME}/scratch/torch_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}"

# ---- Conda environment ----
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ENVS_PATH}/ppo-refusal-env"
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi
conda activate "${ENV_PREFIX}"

echo "[stage] conda env ready: ${ENV_PREFIX}"

echo "[stage] python: $(python --version)"

# ---- Dependencies ----
INSTALL_DEPS="${INSTALL_DEPS:-1}"
if [ "${INSTALL_DEPS}" = "1" ]; then
  echo "[stage] installing/checking python deps (INSTALL_DEPS=1)"
  python -u - <<'PY'
import importlib, os, subprocess, sys

def ensure(import_name, pip_name=None, extra=None):
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
    except Exception:
        cmd = [sys.executable, "-m", "pip", "install"]
        if extra:
            cmd += extra
        cmd += [pip_name]
        subprocess.check_call(cmd)

# Keep pip itself reasonably current
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

# Torch (GPU)
torch_index = os.environ.get("TORCH_INDEX_URL")
extra = ["--index-url", torch_index] if torch_index else None
ensure("torch", "torch", extra=extra)

# Core stack for PPO training
for imp, pip in [
    ("transformers", "transformers>=4.45"),
    ("accelerate", "accelerate>=0.34"),
    ("datasets", "datasets>=2.20"),
    ("peft", "peft>=0.12"),
    ("trl", "trl>=0.11"),
    ("sentencepiece", "sentencepiece"),
    ("pandas", "pandas"),
    ("bitsandbytes", "bitsandbytes"),
    ("sentence_transformers", "sentence-transformers"),
    ("safetensors", "safetensors"),
]:
    ensure(imp, pip)

# Optional: wandb for logging
try:
    import wandb
except Exception:
    print("[deps] wandb not installed, logging disabled")

print("[deps] ok")
PY
  echo "[stage] python deps ready"
else
  echo "[stage] checking python deps (INSTALL_DEPS=0)"
  python -u - <<'PY'
import importlib, sys
need = ["torch", "transformers", "accelerate", "datasets", "peft", "trl",
        "sentencepiece", "pandas", "bitsandbytes", "sentence_transformers", "safetensors"]
missing = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print("[deps] missing:", ", ".join(missing), file=sys.stderr)
    print("[deps] Fix by preparing the conda env on a login/interactive node, or run with INSTALL_DEPS=1.", file=sys.stderr)
    raise SystemExit(2)
print("[deps] ok")
PY
fi

echo "[env] Python: $(python --version)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:"; nvidia-smi || true

# ---- Paths / knobs ----
# Absolute path to repo root on PACE (override with REPO_ROOT env var if needed)
REPO_ROOT="${REPO_ROOT:-${HOME}/scratch/ClaRE}"

# Paths with apostrophes - use simple assignment (works reliably in bash)
SCRIPT_DIR="${REPO_ROOT}/Alec's Stuff/OR_Fine_Tuning/ppo_refusal"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ppo.py"

# Data paths - allow env var override via simple conditional
if [ -z "${DATA_PATH:-}" ]; then
  DATA_PATH="${SCRIPT_DIR}/example_prompts.jsonl"
fi

if [ -z "${REFUSAL_VECTOR_PATH:-}" ]; then
  REFUSAL_VECTOR_PATH="${REPO_ROOT}/Alec's Stuff/Getting_Refusal_Vector/Vector_Extraction/refusal_vector.npz"
fi

LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"  # Leave empty if starting fresh

echo "[paths] REPO_ROOT=${REPO_ROOT}"
echo "[paths] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[paths] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[paths] DATA_PATH=${DATA_PATH}"
echo "[paths] REFUSAL_VECTOR_PATH=${REFUSAL_VECTOR_PATH}"
echo "[paths] LORA_CHECKPOINT=${LORA_CHECKPOINT}"

# Check required files
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[error] Training script not found: ${TRAIN_SCRIPT}" >&2
  echo "Set REPO_ROOT to your ClaRE repo location." >&2
  exit 2
fi

if [ ! -f "${DATA_PATH}" ]; then
  echo "[error] Data file not found: ${DATA_PATH}" >&2
  echo "Set DATA_PATH to your prompts file (JSONL/CSV/TXT)." >&2
  exit 2
fi

if [ ! -f "${REFUSAL_VECTOR_PATH}" ]; then
  echo "[error] Refusal vector not found: ${REFUSAL_VECTOR_PATH}" >&2
  exit 2
fi

# ---- Training configuration ----
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
MODE="${MODE:-increase_refusal}"  # or decrease_refusal
NUM_STEPS="${NUM_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SIMILARITY_WEIGHT="${SIMILARITY_WEIGHT:-0.3}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.7}"
ACTIVATION_LAYER="${ACTIVATION_LAYER:-16}"
SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-0}"  # Set to 1 to enable wandb
WANDB_PROJECT="${WANDB_PROJECT:-refusal-ppo}"

RUN_NAME="ppo_refusal_${MODE}_$(date +%Y%m%d_%H%M%S)"
OUTDIR="${HOME}/scratch/ppo_checkpoints/${RUN_NAME}"
mkdir -p "${OUTDIR}"

echo "[config] MODEL_ID=${MODEL_ID}"
echo "[config] MODE=${MODE}"
echo "[config] NUM_STEPS=${NUM_STEPS}"
echo "[config] BATCH_SIZE=${BATCH_SIZE}"
echo "[config] OUTDIR=${OUTDIR}"

# ---- Build command ----
CMD="python -u \"${TRAIN_SCRIPT}\" \\
  --base_model \"${MODEL_ID}\" \\
  --data_path \"${DATA_PATH}\" \\
  --refusal_vector_path \"${REFUSAL_VECTOR_PATH}\" \\
  --output_dir \"${OUTDIR}\" \\
  --mode ${MODE} \\
  --num_steps ${NUM_STEPS} \\
  --batch_size ${BATCH_SIZE} \\
  --learning_rate ${LEARNING_RATE} \\
  --similarity_weight ${SIMILARITY_WEIGHT} \\
  --similarity_threshold ${SIMILARITY_THRESHOLD} \\
  --activation_layer ${ACTIVATION_LAYER} \\
  --seed ${SEED}"

# Add LoRA checkpoint if specified
if [ -n "${LORA_CHECKPOINT}" ] && [ -d "${LORA_CHECKPOINT}" ]; then
  CMD="${CMD} --lora_checkpoint \"${LORA_CHECKPOINT}\""
fi

# Add wandb if enabled
if [ "${USE_WANDB}" = "1" ]; then
  CMD="${CMD} --use_wandb --wandb_project ${WANDB_PROJECT}"
fi

# ---- Launch ----
echo "[launch] ${CMD}"
eval "${CMD}"

echo "[done] Output dir: ${OUTDIR}"
