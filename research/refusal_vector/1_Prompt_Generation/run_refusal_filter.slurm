#!/bin/bash
#SBATCH --job-name=refusal_filter
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-159%40
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm-%j.out

set -euo pipefail

# —— Caches (optional but recommended) ——
export HF_HOME="$HOME/scratch/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# —— Hugging Face auth ——
export HUGGING_FACE_HUB_TOKEN=

echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi || true

# —— Paths / knobs ——
IN="$HOME/scratch/llama8b_outputs_filtered.csv"            # your prompt,response csv
OUTBASE="$HOME/scratch/refusals_only"                      # shards → OUTBASE.shardNNN.csv
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"                 # use same model you already access
BATCH_SIZE=8
MAX_NEW=4
TEST=0

# —— Conda env ——
module load anaconda3
# Some images need this to make 'conda activate' available
source "$(conda info --base)/etc/profile.d/conda.sh" || eval "$(conda shell.bash hook)"

ENVNAME="llm-refusal-env"
conda env list | awk '{print $1}' | grep -qx "$ENVNAME" || conda create -y -n "$ENVNAME" python=3.10
conda activate "$ENVNAME"

# —— Deps (idempotent) ——
python - << 'PY'
import importlib, subprocess, sys
def sh(*a): subprocess.check_call(list(a))
def ok(m): 
    try: importlib.import_module(m)
    except Exception: return False
    return True

# Torch CUDA 12.x wheels (works on many <cluster> images; adjust if needed)
if not ok("torch"):
    sh(sys.executable, "-m", "pip", "install", "--extra-index-url", "https://download.pytorch.org/whl/cu121", "torch")
for pkg in ("transformers","accelerate","huggingface_hub"):
    if not ok(pkg):
        sh(sys.executable, "-m", "pip", "install", pkg)
PY

# —— Array/sharding ——
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_IDX=${SLURM_ARRAY_TASK_ID:-0}

# —— Run ——
python "$HOME/run_refusal_filter.py" \
  --input  "$IN" \
  --output_base "$OUTBASE" \
  --model  "$MODEL" \
  --num_shards "$NUM_SHARDS" \
  --shard_idx  "$SHARD_IDX" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW" \
  --test "$TEST"