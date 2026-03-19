#!/bin/bash
#SBATCH -J ppo_sweep
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:?CONFIG_FILE must be set}"
SWEEP_NAME="${SWEEP_NAME:-sweep}"

echo "[sweep] job start — config=${CONFIG_FILE} sweep=${SWEEP_NAME}"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

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

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_ENVS_PATH}/ppo-or-env"
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi
conda activate "${ENV_PREFIX}"

DEPS_MARKER="${ENV_PREFIX}/.deps_ok"
if [ ! -f "${DEPS_MARKER}" ]; then
  echo "[stage] installing python deps"
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
print("[deps] ok")
PY
  touch "${DEPS_MARKER}"
fi

echo "[env] Python: $(python --version)"
echo "Node: $(hostname)"
echo "GPU:"; nvidia-smi || true

SCRIPT_DIR="${SCRIPT_DIR:-${SLURM_SUBMIT_DIR}}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_ppo.py"

SCRATCH_DATA="${SCRATCH_DATA:-${HOME}/scratch/ppo_or_data}"
DATA_PATH="${DATA_PATH:-${SCRATCH_DATA}/databricks-dolly-15k.jsonl}"
REFUSAL_VECTOR_PATH="${REFUSAL_VECTOR_PATH:-${SCRATCH_DATA}/refusal_vector.layer032.npz}"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[error] Training script not found: ${TRAIN_SCRIPT}" >&2; exit 2
fi
if [ ! -f "${DATA_PATH}" ]; then
  echo "[error] Data file not found: ${DATA_PATH}" >&2; exit 2
fi
if [ ! -f "${REFUSAL_VECTOR_PATH}" ]; then
  echo "[error] Refusal vector not found: ${REFUSAL_VECTOR_PATH}" >&2; exit 2
fi

RUN_NAME="${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S)"
OUTDIR="${HOME}/scratch/ppo_or_checkpoints/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cp "${CONFIG_FILE}" "${OUTDIR}/sweep_config.json"

python -u - "${CONFIG_FILE}" "${DATA_PATH}" "${REFUSAL_VECTOR_PATH}" "${OUTDIR}" <<'PYSCRIPT'
import json, sys

config_path, data_path, refusal_path, outdir = sys.argv[1:5]
with open(config_path) as f:
    cfg = json.load(f)

cfg.setdefault("training", {})
cfg["training"]["data_path"] = data_path
cfg["training"]["output_dir"] = outdir
cfg.setdefault("model", {})
cfg["model"]["refusal_vector_path"] = refusal_path

patched_path = outdir + "/config_patched.json"
with open(patched_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"[sweep] Patched config written to {patched_path}")
PYSCRIPT

PATCHED_CONFIG="${OUTDIR}/config_patched.json"

echo "[sweep] Config: ${CONFIG_FILE}"
echo "[sweep] Output: ${OUTDIR}"
echo "[sweep] Run name: ${RUN_NAME}"

python -u "${TRAIN_SCRIPT}" --config "${PATCHED_CONFIG}"

echo "[sweep-done] Output dir: ${OUTDIR}"
echo "[sweep-done] Run name: ${RUN_NAME}"
