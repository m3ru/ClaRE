#!/bin/bash
#SBATCH -J llama8b_test
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --array=0-159%40
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

:

# ---- Hugging Face token (PASTE YOUR TOKEN TO THE RIGHT OF =) ----
export HUGGING_FACE_HUB_TOKEN=
export HF_HOME="${HOME}/scratch/hf_cache"
mkdir -p "${HF_HOME}"

# ---- Conda environment (matches previous setup) ----
# You can override this to match your cluster's CUDA version, e.g., cu124 or cpu
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENVNAME=llama8b-env
if ! conda env list | awk '{print $1}' | grep -qx "${ENVNAME}"; then
  conda create -y -n "${ENVNAME}" python=3.10
fi
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
for p in ("transformers", "accelerate", "sentencepiece"):
    ensure(p)
PY

PY="python"
echo "[env] Using Python: $(python --version)"

# ---- Paths / knobs ----
INPUT="${HOME}/scratch/Final-People-Reddit-Prompts-Multiple-Wrappers(in).csv"   # one prompt per line
OUTBASE="${HOME}/scratch/outputs_reddit"     # shards will become OUTBASE.shardNNN.csv
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Array/sharding
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT}
SHARD_IDX=${SLURM_ARRAY_TASK_ID}

# Run mode
TEST=0                     # 1 => run only 10 prompts; 0 => full slice of this shard
BATCH_SIZE=8
MAX_NEW=256
DTYPE=bf16
DO_SAMPLE_FLAG=""          # set to "--do_sample" if you want sampling
TEMPERATURE=0.7
TOP_P=0.9

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:"
nvidia-smi || true

# ---- Launch ----
"${PY}" -u "${HOME}/run_llama_prompts_lines_sharded.py" \
  --input "${INPUT}" \
  --output_base "${OUTBASE}" \
  --model "${MODEL}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}" \
  --batch_size "${BATCH_SIZE}" \
  --max_new_tokens "${MAX_NEW}" \
  --dtype "${DTYPE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --test "${TEST}" \
  ${DO_SAMPLE_FLAG}
