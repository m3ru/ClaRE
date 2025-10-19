#!/bin/bash
#SBATCH -J steer_refusal_demo
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ---- HF token ----
export HUGGING_FACE_HUB_TOKEN=
export HF_HOME="${HOME}/scratch/hf_cache"
mkdir -p "${HF_HOME}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENVNAME=llama8b-env
if ! conda env list | awk '{print $1}' | grep -qx "${ENVNAME}"; then
  conda create -y -n "${ENVNAME}" python=3.10
fi
conda activate "${ENVNAME}"

# Ensure deps
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
for p in ("transformers", "accelerate", "sentencepiece", "numpy"):
    ensure(p)
PY

PY="python"
echo "[env] Using Python: $(python --version)"
nvidia-smi || true

# Inputs (edit as needed)
BENIGN_CSV="${BENIGN_CSV:-${HOME}/scratch/final_benign_prompts.csv}"
VECTOR_NPZ="${VECTOR_NPZ:-${HOME}/scratch/refusal_vector.npz}"  # use multi-layer file by default
SELECT_LAYER="${SELECT_LAYER:-max}"
# Gentler default scales; adjust via Job Composer env if desired
SCALES="${SCALES:-0,0.25,0.5,1,4}"
CALIBRATION_ALPHA="${CALIBRATION_ALPHA:-0.2}"
NUM_PROMPTS=${NUM_PROMPTS:-10}
OUTCSV="${OUTCSV:-${HOME}/scratch/steer_refusal_demo.csv}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

${PY} -u "${HOME}/steer_refusal_demo.py" \
  --benign_csv "${BENIGN_CSV}" \
  --vector_npz "${VECTOR_NPZ}" \
  --select_layer "${SELECT_LAYER}" \
  --scales "${SCALES}" \
  --calibration_alpha "${CALIBRATION_ALPHA}" \
  --num_prompts "${NUM_PROMPTS}" \
  --output_csv "${OUTCSV}" \
  --model "${MODEL}"

echo "Wrote ${OUTCSV}"


