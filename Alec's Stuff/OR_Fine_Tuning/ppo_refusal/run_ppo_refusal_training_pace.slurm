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

# ---- Hugging Face token ----
# Option 1: Set token directly here (hardcode):
#   export HUGGING_FACE_HUB_TOKEN="hf_..."
# NOTE: This is sensitive. Don't commit/share this file if you hardcode a token.
# Paste your HF token here, or run `huggingface-cli login` on an interactive node
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
# Option 2: Set in ~/.bashrc or use huggingface-cli login

# Fallback: try to read from huggingface-cli cache if not set
if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
  echo "[auth] Loaded HF token from ~/.cache/huggingface/token"
fi

# Debug: verify token is available
if [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "[auth] HF_TOKEN is set: yes (starts with: ${HUGGING_FACE_HUB_TOKEN:0:10}...)"
else
  echo "[auth] WARNING: No HF token found! Gated models will fail."
  echo "[auth] Run 'huggingface-cli login' on an interactive node first."
fi

# ---- Put conda/pip caches + envs in SCRATCH (avoid HOME quota) ----
export SCRATCH_BASE="${HOME}/scratch"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda_pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda_envs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${TMPDIR}"

# ---- Caches (PACE scratch recommended) ----
export HF_HOME="${HOME}/scratch/hf_cache"
# Avoid duplicate caches: TRANSFORMERS_CACHE is deprecated and can create a second cache tree.
# HF uses ${HF_HOME}/hub by default; set HF_HUB_CACHE explicitly.
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HOME}/scratch/torch_cache"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}"

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
# Installing inside every job will repeatedly download wheels and can hit quota.
# We install ONCE per scratch conda env, then reuse.
DEPS_MARKER="${ENV_PREFIX}/.deps_ok"
if [ -f "${DEPS_MARKER}" ]; then
  INSTALL_DEPS="${INSTALL_DEPS:-0}"
else
  INSTALL_DEPS="${INSTALL_DEPS:-1}"
fi
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

# Keep pip itself reasonably current (only happens on first install due to marker)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

# Torch (GPU)
torch_index = os.environ.get("TORCH_INDEX_URL")
extra = ["--index-url", torch_index] if torch_index else None
ensure("torch", "torch", extra=extra)

# Pin TRL to the old PPOTrainer API range (avoid force-reinstall to reduce churn)
subprocess.check_call([sys.executable, "-m", "pip", "install", "trl>=0.9.0,<0.12.0"])

# Core stack for PPO training
for imp, pip in [
    ("transformers", "transformers>=4.45"),
    ("accelerate", "accelerate>=0.34"),
    ("datasets", "datasets>=2.20"),
    ("peft", "peft>=0.12"),
    ("trl", "trl>=0.9.0,<0.12.0"),  # Pin to old API (0.12+ has breaking changes)
    ("sentencepiece", "sentencepiece"),
    ("pandas", "pandas"),
    ("bitsandbytes", "bitsandbytes"),
    ("sentence_transformers", "sentence-transformers"),
    ("safetensors", "safetensors"),
    ("rich", "rich"),  # Required by TRL
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
  # Mark env as ready so subsequent jobs reuse without reinstalling
  touch "${DEPS_MARKER}"
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
# You said you do NOT want to upload the whole repo.
# Put everything needed for PPO into a single scratch folder:
#   ~/scratch/clare_ppo
#
# Expected contents:
#   ~/scratch/clare_ppo/ppo_refusal/  (this folder: train_ppo.py, config.py, data.py, reward_model.py, etc.)
#   ~/scratch/clare_ppo/OR_LoRA_Fine-Tuning_Results/  (your SFT LoRA adapter folder)
#   ~/scratch/clare_ppo/refusal_vector.npz  (refusal vector)
#   ~/scratch/clare_ppo/prompts.jsonl  (your PPO training prompts; JSONL/CSV/TXT supported)
CLARE_PPO_DIR="${CLARE_PPO_DIR:-${HOME}/scratch/clare_ppo}"
SCRIPT_DIR="${CLARE_PPO_DIR}/ppo_refusal"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ppo.py"

# Data paths - allow env var override
if [ -z "${DATA_PATH:-}" ]; then
  DATA_PATH="${CLARE_PPO_DIR}/databricks-dolly-15k.jsonl"
fi

if [ -z "${REFUSAL_VECTOR_PATH:-}" ]; then
  REFUSAL_VECTOR_PATH="${CLARE_PPO_DIR}/refusal_vector.npz"
fi

# Optional override; otherwise train_ppo.py has an internal default:
#   ~/scratch/clare_ppo/OR_LoRA_Fine-Tuning_Results
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"

echo "[paths] CLARE_PPO_DIR=${CLARE_PPO_DIR}"
echo "[paths] SCRIPT_DIR=${SCRIPT_DIR}"
echo "[paths] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[paths] DATA_PATH=${DATA_PATH}"
echo "[paths] REFUSAL_VECTOR_PATH=${REFUSAL_VECTOR_PATH}"
echo "[paths] LORA_CHECKPOINT=${LORA_CHECKPOINT}"

# Check required files
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[error] Training script not found: ${TRAIN_SCRIPT}" >&2
  echo "Copy the ppo_refusal folder into: ${CLARE_PPO_DIR}/ppo_refusal" >&2
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
ACTIVATION_LAYER="${ACTIVATION_LAYER:-32}"
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
