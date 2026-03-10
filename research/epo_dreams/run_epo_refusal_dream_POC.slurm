#!/bin/bash
#SBATCH -J epo_refusal_dream
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ---- Hugging Face token ----
export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
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

# ---- Inputs/knobs ----
VECTOR_NPZ="${VECTOR_NPZ:-${HOME}/scratch/refusal_vector.npz}"
SELECT_LAYER="${SELECT_LAYER:-max}"
LAST_K="${LAST_K:-1}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DTYPE="${DTYPE:-bf16}"
POP="${POP:-4}"
STEPS="${STEPS:-300}"
TOPK="${TOPK:-64}"
MAX_USER_TOKENS="${MAX_USER_TOKENS:-48}"
SEEDS_CSV="${SEEDS_CSV:-}"  # optional benign seeds
NUM_SEEDS="${NUM_SEEDS:-12}"
LAMBDA_MIN="${LAMBDA_MIN:-0.0}"
LAMBDA_MAX="${LAMBDA_MAX:-2.0}"
FEATURE_WEIGHT="${FEATURE_WEIGHT:-3.0}"
OUTCSV="${OUTCSV:-${HOME}/scratch/epo_refusal_prompts.csv}"
TEST="${TEST:-0}"

"${PY}" -u "${HOME}/epo_refusal_dream.py" \
  --vector_npz "${VECTOR_NPZ}" \
  --select_layer "${SELECT_LAYER}" \
  --last_k "${LAST_K}" \
  --model "${MODEL}" \
  --dtype "${DTYPE}" \
  --population "${POP}" \
  --steps_per_member "${STEPS}" \
  --topk_per_pos "${TOPK}" \
  --max_user_tokens "${MAX_USER_TOKENS}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_seeds "${NUM_SEEDS}" \
  --lambda_min "${LAMBDA_MIN}" \
  --lambda_max "${LAMBDA_MAX}" \
  --feature_weight "${FEATURE_WEIGHT}" \
  --output_csv "${OUTCSV}" \
  --test "${TEST}"

echo "Saved prompts to ${OUTCSV}"


