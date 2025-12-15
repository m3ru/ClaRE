#!/bin/bash
#SBATCH -J or_test_rewriter
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

echo "[stage] job start"

# ---- Hugging Face token (paste here if you want it hardcoded) ----
export HUGGING_FACE_HUB_TOKEN=

# ---- Put conda/pip caches + envs in SCRATCH (avoid HOME quota) ----
export SCRATCH_BASE="${HOME}/scratch"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda_pkgs"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda_envs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${TMPDIR}"

# ---- HF caches ----
export HF_HOME="${HOME}/scratch/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HOME}/scratch/torch_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}"

# ---- Conda env ----
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

# ---- Dependencies (idempotent) ----
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

for imp, pip in [
    ("transformers", "transformers>=4.45"),
    ("accelerate", "accelerate>=0.34"),
    ("peft", "peft>=0.12"),
    ("sentencepiece", "sentencepiece"),
]:
    ensure(imp, pip)

print("[deps] ok")
PY

echo "[env] Python: $(python --version)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:"; nvidia-smi || true

# ---- Paths ----
# Put this test script + an optional test_prompts.txt into a scratch folder.
DATA_DIR="${DATA_DIR:-${HOME}/scratch/clare_or_ft}"
SCRIPT_PATH="${DATA_DIR}/test_overrefusal_prompt_rewriter.py"
PROMPTS_FILE="${PROMPTS_FILE:-${DATA_DIR}/test_prompts.txt}"

ADAPTER_DIR="${ADAPTER_DIR:-/home/hice1/aharris345/scratch/or_finetunes/or_prompt_rewriter_20251214_133618}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

OUTCSV="${OUTCSV:-${ADAPTER_DIR}/test_rewrites.csv}"

echo "[paths] SCRIPT_PATH=${SCRIPT_PATH}"
echo "[paths] PROMPTS_FILE=${PROMPTS_FILE}"
echo "[paths] ADAPTER_DIR=${ADAPTER_DIR}"
echo "[paths] OUTCSV=${OUTCSV}"

if [ ! -f "${SCRIPT_PATH}" ]; then
  echo "[error] Missing test script: ${SCRIPT_PATH}" >&2
  exit 2
fi

# If prompts file doesn't exist, we'll pass a few default prompts via args.
DEFAULT_PROMPTS=(
  "Explain how photosynthesis works."
  "What are common causes of headaches and when should someone see a doctor?"
  "How do I write a polite email asking for an extension on a deadline?"
)

if [ -f "${PROMPTS_FILE}" ]; then
  python -u "${SCRIPT_PATH}" \
    --base_model "${BASE_MODEL}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --prompts_file "${PROMPTS_FILE}" \
    --output_csv "${OUTCSV}" \
    --max_new_tokens 200 \
    --do_sample 0
else
  echo "[warn] PROMPTS_FILE not found; using built-in defaults." >&2
  CMD=(python -u "${SCRIPT_PATH}" --base_model "${BASE_MODEL}" --adapter_dir "${ADAPTER_DIR}" --output_csv "${OUTCSV}" --max_new_tokens 200 --do_sample 0)
  for p in "${DEFAULT_PROMPTS[@]}"; do
    CMD+=(--prompt "$p")
  done
  "${CMD[@]}"
fi

echo "[done] Wrote ${OUTCSV}"
