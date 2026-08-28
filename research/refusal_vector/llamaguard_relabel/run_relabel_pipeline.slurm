#!/bin/bash
#SBATCH -J lg_relabel
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Relabel the original prompt pool by Llama-Guard's OWN safe/unsafe verdict,
# then sweep all layers by separation AUC to pick the best 'unsafe' direction.
# Knobs (env): MAX_PROMPTS (0=all), DO_SWEEP (1/0), N_PER_CLASS.

set -euo pipefail
echo "[stage] lg_relabel start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- HF token (Llama-Guard is gated) ----
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
fi
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
[ -n "${HUGGING_FACE_HUB_TOKEN}" ] || { echo "[error] no HF token"; exit 2; }

# ---- caches: READ model from scratch (already cached), WRITE nothing there ----
# Scratch is chronically at 100% from other jobs, so keep all writes on home.
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"        # model hub lives here (read-only use)
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1                            # load cached model, no ref/lock writes
export HF_DATASETS_CACHE="${HOME}/.cache/hf_datasets_relabel"
export TMPDIR="${HOME}/tmp_relabel"
export HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
echo "[stage] python: $(python --version)"
nvidia-smi | head -15 || true

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/anon-repo}"
DIR="${PROJECT_ROOT}/research/refusal_vector/llamaguard_relabel"
# unique_prompts.csv = prompt column extracted from llama8b_outputs_filtered.csv
# (the exact original pool; small, ships with the relabel dir).
INPUT="${INPUT:-${DIR}/unique_prompts.csv}"
[ -f "${INPUT}" ] || { echo "[error] missing input prompts: ${INPUT}"; exit 2; }
cd "${DIR}"

MAX_PROMPTS="${MAX_PROMPTS:-0}"
DO_SWEEP="${DO_SWEEP:-1}"
N_PER_CLASS="${N_PER_CLASS:-2000}"
MODEL="${MODEL:-meta-llama/Llama-Guard-3-8B}"

echo "[config] input=${INPUT} max_prompts=${MAX_PROMPTS} do_sweep=${DO_SWEEP}"

# ---- Step 1: label by Llama-Guard verdict ----
python -u label_prompts_llamaguard.py \
  --input_csv "${INPUT}" --prompt_col prompt \
  --output_csv "${DIR}/llamaguard_labels.csv" \
  --unsafe_out "${DIR}/lg_unsafe_prompts.csv" \
  --safe_out "${DIR}/lg_safe_prompts.csv" \
  --model "${MODEL}" --batch_size 16 --max_length 512 --max_prompts "${MAX_PROMPTS}"

# ---- Step 2: layer-separation sweep + save best vector ----
if [ "${DO_SWEEP}" = "1" ]; then
  python -u sweep_layers.py \
    --unsafe_csv "${DIR}/lg_unsafe_prompts.csv" \
    --safe_csv "${DIR}/lg_safe_prompts.csv" \
    --model "${MODEL}" --n_per_class "${N_PER_CLASS}" \
    --val_frac 0.3 --max_length 512 --batch_size 16 \
    --output_dir "${DIR}"
fi

echo "[done] relabel pipeline complete at $(date)"
ls -la "${DIR}"/*.npz "${DIR}"/*.json 2>&1 || true
