#!/bin/bash
#SBATCH -J pez_opt
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --array=0-9%10
#SBATCH --output=slurm-%x-%A-%a.out
#SBATCH --error=slurm-%x-%A-%a.err

set -euo pipefail

# ============================================================================
# PEZ Optimization for Refusal Vector Prompt Discovery
# ============================================================================
# This script runs PEZ (Prompt Embedding Zeroization) to discover prompts
# with specific refusal activation characteristics.
#
# Usage:
#   sbatch --array=0-9%10 run_pez_optimization.slurm
#   sbatch --array=0-9%10 --export=OBJECTIVE=maximize run_pez_optimization.slurm
#   sbatch --array=0-9%10 --export=OBJECTIVE=target,TARGET_ACTIVATION=0.5 run_pez_optimization.slurm
# ============================================================================

# ---- Fill in your HF token ----

export HF_HOME="${HOME}/scratch/hf_cache"
mkdir -p "${HF_HOME}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENVNAME=llama8b-env

# Clean corrupted conda cache if needed (common on shared filesystems)
if ! conda env list | awk '{print $1}' | grep -qx "${ENVNAME}"; then
  echo "Creating new conda environment: ${ENVNAME}"

  # Try to clean conda cache first to avoid corruption issues
  conda clean -y --all 2>/dev/null || true

  # Attempt to create environment
  if ! conda create -y -n "${ENVNAME}" python=3.10 2>&1; then
    echo "First attempt failed, cleaning cache and retrying..."
    # Force remove corrupted packages
    rm -rf "${HOME}/.conda/pkgs/ncurses-"* 2>/dev/null || true
    rm -rf "${HOME}/.conda/pkgs/openssl-"* 2>/dev/null || true
    rm -rf "${HOME}/.conda/pkgs/python-"* 2>/dev/null || true
    rm -rf "${HOME}/.conda/pkgs/pip-"* 2>/dev/null || true
    rm -rf "${HOME}/.conda/pkgs/setuptools-"* 2>/dev/null || true
    rm -rf "${HOME}/.conda/pkgs/tk-"* 2>/dev/null || true

    # Clean conda cache again
    conda clean -y --all 2>/dev/null || true

    # Retry environment creation
    conda create -y -n "${ENVNAME}" python=3.10
  fi
fi

conda activate "${ENVNAME}"

# Ensure dependencies
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

# Additional deps for PEZ
try:
    import matplotlib
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
PY

PY="python"
echo "[env] Using Python: $(python --version)"
nvidia-smi || true

# ============================================================================
# Configuration
# ============================================================================

# Model settings
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DTYPE="${DTYPE:-bf16}"

# Refusal vector settings
REFUSAL_VECTOR="${REFUSAL_VECTOR:-${HOME}/scratch/refusal_vector.npz}"
LAYERS="${LAYERS:-all}"  # "all", "max", or comma-separated like "8,16,24"

# Optimization settings
NUM_STEPS="${NUM_STEPS:-3000}"
LEARNING_RATE="${LEARNING_RATE:-0.1}"
PROMPT_LENGTH="${PROMPT_LENGTH:-20}"

# Objective settings - can be set via --export when submitting
OBJECTIVE="${OBJECTIVE:-target}"  # maximize, minimize, target, boundary
TARGET_ACTIVATION="${TARGET_ACTIVATION:-0.0}"
BOUNDARY_MARGIN="${BOUNDARY_MARGIN:-0.1}"

# Initialization settings
INIT_MODE="${INIT_MODE:-random}"  # random, seed, mixed
SEED_PROMPT="${SEED_PROMPT:-}"
NUM_RANDOM_TOKENS="${NUM_RANDOM_TOKENS:-10}"

# Optimizer settings
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRADIENT_CLIP="${GRADIENT_CLIP:-1.0}"

# Output settings
SAVE_EVERY="${SAVE_EVERY:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/scratch/pez_results}"

# Compute settings
DEVICE="${DEVICE:-cuda}"

# Use array task ID as seed for reproducibility
SEED="${SLURM_ARRAY_TASK_ID:-42}"

# Auto-generate run name based on objective and array ID
if [[ -z "${RUN_NAME:-}" ]]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RUN_NAME="pez_${OBJECTIVE}_${INIT_MODE}_seed${SEED}_${TIMESTAMP}"
fi

# ============================================================================
# Print Configuration
# ============================================================================

echo "============================================================================"
echo "PEZ Optimization Configuration"
echo "============================================================================"
echo "Model:               ${MODEL}"
echo "Refusal Vector:      ${REFUSAL_VECTOR}"
echo "Layers:              ${LAYERS}"
echo "Objective:           ${OBJECTIVE}"
if [[ "${OBJECTIVE}" == "target" ]]; then
  echo "Target Activation:   ${TARGET_ACTIVATION}"
fi
if [[ "${OBJECTIVE}" == "boundary" ]]; then
  echo "Boundary Margin:     ${BOUNDARY_MARGIN}"
fi
echo "Initialization:      ${INIT_MODE}"
if [[ "${INIT_MODE}" == "seed" || "${INIT_MODE}" == "mixed" ]]; then
  echo "Seed Prompt:         ${SEED_PROMPT}"
fi
echo "Prompt Length:       ${PROMPT_LENGTH}"
echo "Steps:               ${NUM_STEPS}"
echo "Learning Rate:       ${LEARNING_RATE}"
echo "Optimizer:           ${OPTIMIZER}"
echo "Random Seed:         ${SEED}"
echo "Run Name:            ${RUN_NAME}"
echo "Output Dir:          ${OUTPUT_DIR}"
echo "============================================================================"
echo ""

# ============================================================================
# Run PEZ Optimization
# ============================================================================

# Determine script location (same directory as this SLURM script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEZ_SCRIPT="${SCRIPT_DIR}/pez_refusal_optimization.py"

# Verify script exists
if [[ ! -f "${PEZ_SCRIPT}" ]]; then
  echo "ERROR: PEZ script not found at: ${PEZ_SCRIPT}"
  echo "Please ensure pez_refusal_optimization.py is in the same directory as this SLURM script"
  exit 1
fi

echo "Using PEZ script: ${PEZ_SCRIPT}"
echo ""

# Build command with all parameters
CMD="${PY} -u \"${PEZ_SCRIPT}\" \
  --model \"${MODEL}\" \
  --dtype \"${DTYPE}\" \
  --refusal_vector \"${REFUSAL_VECTOR}\" \
  --layers \"${LAYERS}\" \
  --num_steps \"${NUM_STEPS}\" \
  --learning_rate \"${LEARNING_RATE}\" \
  --prompt_length \"${PROMPT_LENGTH}\" \
  --objective \"${OBJECTIVE}\" \
  --target_activation \"${TARGET_ACTIVATION}\" \
  --boundary_margin \"${BOUNDARY_MARGIN}\" \
  --init_mode \"${INIT_MODE}\" \
  --num_random_tokens \"${NUM_RANDOM_TOKENS}\" \
  --optimizer \"${OPTIMIZER}\" \
  --weight_decay \"${WEIGHT_DECAY}\" \
  --gradient_clip \"${GRADIENT_CLIP}\" \
  --save_every \"${SAVE_EVERY}\" \
  --output_dir \"${OUTPUT_DIR}\" \
  --run_name \"${RUN_NAME}\" \
  --device \"${DEVICE}\" \
  --seed \"${SEED}\""

# Add seed prompt if provided
if [[ -n "${SEED_PROMPT}" ]]; then
  CMD="${CMD} --seed_prompt \"${SEED_PROMPT}\""
fi

echo "Executing PEZ optimization..."
echo ""

eval ${CMD}

echo ""
echo "============================================================================"
echo "PEZ Optimization Complete"
echo "============================================================================"
echo "Results saved to: ${OUTPUT_DIR}/${RUN_NAME}"
echo "============================================================================"
