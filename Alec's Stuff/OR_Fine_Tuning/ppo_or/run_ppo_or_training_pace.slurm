#!/bin/bash
#SBATCH -J ppo_or
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
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- Hugging Face token ----
# Paste your token here:
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
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ENVS_PATH}/ppo-or-env"
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi
conda activate "${ENV_PREFIX}"

echo "[stage] conda env ready: ${ENV_PREFIX}"
echo "[stage] python: $(python --version)"

# ---- Dependencies ----
DEPS_MARKER="${ENV_PREFIX}/.deps_ok"
if [ -f "${DEPS_MARKER}" ]; then
  INSTALL_DEPS="${INSTALL_DEPS:-0}"
else
  INSTALL_DEPS="${INSTALL_DEPS:-1}"
fi
if [ "${INSTALL_DEPS}" = "1" ]; then
  echo "[stage] installing python deps (INSTALL_DEPS=1)"
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

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

torch_index = os.environ.get("TORCH_INDEX_URL")
extra = ["--index-url", torch_index] if torch_index else None
ensure("torch", "torch", extra=extra)

subprocess.check_call([sys.executable, "-m", "pip", "install", "trl>=0.9.0,<0.12.0"])

for imp, pip in [
    ("transformers", "transformers>=4.45"),
    ("accelerate", "accelerate>=0.34"),
    ("datasets", "datasets>=2.20"),
    ("peft", "peft>=0.12"),
    ("trl", "trl>=0.9.0,<0.12.0"),
    ("sentencepiece", "sentencepiece"),
    ("pandas", "pandas"),
    ("bitsandbytes", "bitsandbytes"),
    ("sentence_transformers", "sentence-transformers"),
    ("safetensors", "safetensors"),
    ("rich", "rich"),
]:
    ensure(imp, pip)

try:
    import wandb
except Exception:
    print("[deps] wandb not installed, logging disabled")

print("[deps] ok")
PY
  echo "[stage] python deps ready"
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
    raise SystemExit(2)
print("[deps] ok")
PY
fi

echo "[env] Python: $(python --version)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:"; nvidia-smi || true

# ---- Paths ----
# Code lives in the repo (this directory); data/artifacts live on scratch.
#
# Code (in repo — do NOT put on scratch):
#   ppo_or/train_ppo.py, config.py, data.py, reward_model.py
#
# Data & artifacts (on scratch):
#   ~/scratch/ppo_or_data/databricks-dolly-15k.jsonl
#   ~/scratch/ppo_or_data/refusal_vector.npz
#   ~/scratch/ppo_or_data/OR_LoRA_Fine-Tuning_Results/  (optional SFT checkpoint)
#
# Checkpoints written to:
#   ~/scratch/ppo_or_checkpoints/<run_name>/

# SLURM copies the batch script to a spool dir, so BASH_SOURCE won't work.
# SLURM_SUBMIT_DIR is the directory from which `sbatch` was invoked.
# Run sbatch from inside the ppo_or/ folder, or set SCRIPT_DIR explicitly.
SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR}}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ppo.py"

SCRATCH_DATA="${SCRATCH_DATA:-${HOME}/scratch/ppo_or_data}"
DATA_PATH="${DATA_PATH:-${SCRATCH_DATA}/databricks-dolly-15k.jsonl}"
REFUSAL_VECTOR_PATH="${REFUSAL_VECTOR_PATH:-${SCRATCH_DATA}/refusal_vector.layer032.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-}"

echo "[paths] SCRIPT_DIR=${SCRIPT_DIR}  (code — in repo)"
echo "[paths] TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "[paths] SCRATCH_DATA=${SCRATCH_DATA}  (data — on scratch)"
echo "[paths] DATA_PATH=${DATA_PATH}"
echo "[paths] REFUSAL_VECTOR_PATH=${REFUSAL_VECTOR_PATH}"
echo "[paths] LORA_CHECKPOINT=${LORA_CHECKPOINT}"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[error] Training script not found: ${TRAIN_SCRIPT}" >&2
  exit 2
fi
if [ ! -f "${DATA_PATH}" ]; then
  echo "[error] Data file not found: ${DATA_PATH}" >&2
  echo "Place it at: ${DATA_PATH}" >&2
  exit 2
fi
if [ ! -f "${REFUSAL_VECTOR_PATH}" ]; then
  echo "[error] Refusal vector not found: ${REFUSAL_VECTOR_PATH}" >&2
  echo "Place it at: ${REFUSAL_VECTOR_PATH}" >&2
  exit 2
fi

# ---- Training configuration ----
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
NUM_STEPS="${NUM_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
ACTIVATION_LAYER="${ACTIVATION_LAYER:-32}"
SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-or-ppo}"

RUN_NAME="ppo_or_$(date +%Y%m%d_%H%M%S)"
OUTDIR="${HOME}/scratch/ppo_or_checkpoints/${RUN_NAME}"
mkdir -p "${OUTDIR}"

echo "[config] MODEL_ID=${MODEL_ID}"
echo "[config] NUM_STEPS=${NUM_STEPS}"
echo "[config] BATCH_SIZE=${BATCH_SIZE}"
echo "[config] OUTDIR=${OUTDIR}"

# ---- Build command ----
CMD="python -u \"${TRAIN_SCRIPT}\" \
  --base_model \"${MODEL_ID}\" \
  --data_path \"${DATA_PATH}\" \
  --refusal_vector_path \"${REFUSAL_VECTOR_PATH}\" \
  --output_dir \"${OUTDIR}\" \
  --num_steps ${NUM_STEPS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --activation_layer ${ACTIVATION_LAYER} \
  --seed ${SEED}"

if [ -n "${LORA_CHECKPOINT}" ] && [ -d "${LORA_CHECKPOINT}" ]; then
  CMD="${CMD} --lora_checkpoint \"${LORA_CHECKPOINT}\""
fi

if [ "${USE_WANDB}" = "1" ]; then
  CMD="${CMD} --use_wandb --wandb_project ${WANDB_PROJECT}"
fi

# ---- Launch ----
echo "[launch] ${CMD}"
eval "${CMD}"

echo "[done] Output dir: ${OUTDIR}"
