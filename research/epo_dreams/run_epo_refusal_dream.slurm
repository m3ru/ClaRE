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

# ---- Hugging Face token (set locally; do NOT commit real token) ----
export HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"
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
SEEDS_CSV="${SEEDS_CSV:-${HOME}/benign_seed_prompts.csv}"  # benign seeds to edit
NUM_SEEDS="${NUM_SEEDS:-4}"
LAMBDA_MIN="${LAMBDA_MIN:-0.0}"
LAMBDA_MAX="${LAMBDA_MAX:-0.0}"   # disable CE coherence by default when using semantic similarity
FEATURE_WEIGHT="${FEATURE_WEIGHT:-2.0}"
EMBED_MODEL="${EMBED_MODEL:-nvidia/llama-embed-nemotron-8b}"
EMBED_DTYPE="${EMBED_DTYPE:-bf16}"
SIM_WEIGHT="${SIM_WEIGHT:-1.0}"
GUARD_MODEL="${GUARD_MODEL:-meta-llama/Llama-Guard-3-8B}"
GUARD_DTYPE="${GUARD_DTYPE:-bf16}"
GUARD_WEIGHT="${GUARD_WEIGHT:-1.0}"
GUARD_LABEL_INDEX="${GUARD_LABEL_INDEX:-1}"
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
  --embed_model "${EMBED_MODEL}" \
  --embed_dtype "${EMBED_DTYPE}" \
  --sim_weight "${SIM_WEIGHT}" \
  --guard_model "${GUARD_MODEL}" \
  --guard_dtype "${GUARD_DTYPE}" \
  --guard_weight "${GUARD_WEIGHT}" \
  --guard_label_index "${GUARD_LABEL_INDEX}" \
  --output_csv "${OUTCSV}" \
  --test "${TEST}"

echo "Saved prompts to ${OUTCSV}"


