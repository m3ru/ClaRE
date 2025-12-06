#!/bin/bash
#SBATCH -J phtest_sanitize_oss120b
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH -t 12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Optional: make sure we run from the directory where you invoked sbatch
cd "${SLURM_SUBMIT_DIR}"

# ---- Hugging Face token (set locally; do NOT commit real token) ----
export HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"
export HF_HOME="${HOME}/scratch/hf_cache"
mkdir -p "${HF_HOME}"

# ---- Conda environment (similar to llama job) ----
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

# Re-use the existing environment from your llama job to avoid
# creating a new env and hitting disk quota issues.
ENVNAME=llama8b-env
conda activate "${ENVNAME}"

# Ensure required packages are installed (idempotent)
python - <<'PY'
import importlib, os, subprocess, sys

def ensure(pkg, *extra):
    name = pkg.split("==")[0]
    try:
        importlib.import_module(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *extra, pkg])

torch_index = os.environ.get("TORCH_INDEX_URL")
extra = ["--index-url", torch_index] if torch_index else []
ensure("torch", *extra)
for p in ("transformers", "accelerate", "sentencepiece", "kernels", "datasets", "pandas"):
    ensure(p)
PY

PYTHON="python"
echo "[env] Using Python: $(python --version)"

# ---- Paths / knobs ----
# On PACE you're mirroring the llama setup by placing the Python
# script directly in $HOME, so point to it there.
SCRIPT_PATH="${HOME}/scratch/ClaRE/Alec's Stuff/OR_Fine_Tuning/sanitize_phtest_prompts.py"
OUTPUT_PATH="${HOME}/scratch/phtest_PHTest_train_sanitized.csv"
PROMPT_COLUMN="Request"   # adjust if PHTest uses a different field name
HF_SPLIT="train"         # adjust to the desired split
DEVICE_MAP="auto"        # 'auto' lets transformers spread across available GPU/CPU
MAX_NEW=256

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:"
nvidia-smi || true

# ---- Launch ----
"${PYTHON}" -u "${SCRIPT_PATH}" \
  --use_hf \
  --hf_split "${HF_SPLIT}" \
  --prompt_column "${PROMPT_COLUMN}" \
  --output_path "${OUTPUT_PATH}" \
  --device_map "${DEVICE_MAP}" \
  --max_new_tokens "${MAX_NEW}"