#!/bin/bash
#SBATCH -J or_sft_llama8b
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

# ---- Conda environment (mirrors your generation jobs) ----
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ENVS_PATH}/or-ft-llama8b-env"
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi
conda activate "${ENV_PREFIX}"

echo "[stage] conda env ready: ${ENV_PREFIX}"

echo "[stage] python: $(python --version)"

# ---- Dependencies ----
# You requested fully self-contained jobs (no manual env prep).
# This block installs missing deps inside the job (idempotent).
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

# Keep pip itself reasonably current (helps with many wheels)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

# Torch (GPU)
torch_index = os.environ.get("TORCH_INDEX_URL")
extra = ["--index-url", torch_index] if torch_index else None
ensure("torch", "torch", extra=extra)

# Core stack (no bitsandbytes needed unless you use --use_4bit 1)
for imp, pip in [
    ("transformers", "transformers>=4.45"),
    ("accelerate", "accelerate>=0.34"),
    ("datasets", "datasets>=2.20"),
    ("peft", "peft>=0.12"),
    ("trl", "trl>=0.11"),
    ("sentencepiece", "sentencepiece"),
    ("pandas", "pandas"),
]:
    ensure(imp, pip)
print("[deps] ok")
PY
  echo "[stage] python deps ready"
else
  echo "[stage] checking python deps (INSTALL_DEPS=0)"
  python -u - <<'PY'
import importlib, sys
need = ["torch", "transformers", "accelerate", "datasets", "peft", "trl", "sentencepiece", "pandas"]
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
# You said you are NOT uploading the whole repo to PACE.
# Put ONLY these two files into a scratch folder:
# - PHTest_train_sanitized_cleaned.csv
# - finetune_overrefusal_prompt_rewriter.py
#
# You can override DATA_DIR when submitting:
#   DATA_DIR=$HOME/scratch/somewhere sbatch run_finetune_overrefusal_prompt_rewriter_pace.slurm
DATA_DIR="${DATA_DIR:-${HOME}/scratch/clare_or_ft}"
CSV_PATH="${DATA_DIR}/PHTest_train_sanitized_cleaned.csv"
SCRIPT_PATH="${DATA_DIR}/finetune_overrefusal_prompt_rewriter.py"

echo "[paths] DATA_DIR=${DATA_DIR}"
echo "[paths] CSV_PATH=${CSV_PATH}"
echo "[paths] SCRIPT_PATH=${SCRIPT_PATH}"

if [ ! -f "${CSV_PATH}" ]; then
  echo "[error] CSV not found: ${CSV_PATH}" >&2
  echo "Copy PHTest_train_sanitized_cleaned.csv into ${DATA_DIR} (or set DATA_DIR)." >&2
  exit 2
fi

if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "[error] Training script not found: ${SCRIPT_PATH}" >&2
  echo "Copy finetune_overrefusal_prompt_rewriter.py into ${DATA_DIR} (or set DATA_DIR)." >&2
  exit 2
fi

MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"  # change to your llama-8b

RUN_NAME="or_prompt_rewriter_$(date +%Y%m%d_%H%M%S)"
OUTDIR="${HOME}/scratch/or_finetunes/${RUN_NAME}"
mkdir -p "${OUTDIR}"

# ---- Launch ----
python -u "${SCRIPT_PATH}" \
  --csv "${CSV_PATH}" \
  --model_id "${MODEL_ID}" \
  --output_dir "${OUTDIR}" \
  --max_seq_len 512 \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --eval_steps 200 \
  --save_steps 200 \
  --packing 1 \
  --use_4bit 0

echo "[done] Output dir: ${OUTDIR}"
