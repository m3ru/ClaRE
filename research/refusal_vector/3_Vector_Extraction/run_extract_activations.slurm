#!/bin/bash
#SBATCH -J extract_acts
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --array=0-159%40
#SBATCH --output=slurm-%x-%A-%a.out
#SBATCH --error=slurm-%x-%A-%a.err

set -euo pipefail

# ---- Fill in your HF token ----
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

# ---- Inputs ----
# Set DATASET to either 'benign' or 'refusal' and update INPUT accordingly
DATASET=${DATASET:-refusal}
if [[ "${DATASET}" == "benign" ]]; then
  INPUT="${HOME}/scratch/final_benign_prompts.csv"
  OUTBASE="${HOME}/scratch/acts_benign/acts"
elif [[ "${DATASET}" == "refusal" ]]; then
  INPUT="${HOME}/scratch/final_refusals_prompts.csv"
  OUTBASE="${HOME}/scratch/acts_refusal/acts"
else
  echo "Unknown DATASET='${DATASET}' (use benign|refusal)" >&2; exit 2
fi

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT}
SHARD_IDX=${SLURM_ARRAY_TASK_ID}

BATCH_SIZE=${BATCH_SIZE:-8}
DTYPE=${DTYPE:-bf16}
LAST_K=${LAST_K:-1}
LAYERS=${LAYERS:-all}
TEST=${TEST:-0}

${PY} -u "${HOME}/extract_activations_sharded.py" \
  --input "${INPUT}" \
  --output_base "${OUTBASE}" \
  --model "${MODEL}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}" \
  --batch_size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --last_k "${LAST_K}" \
  --layers "${LAYERS}" \
  --test "${TEST}"


