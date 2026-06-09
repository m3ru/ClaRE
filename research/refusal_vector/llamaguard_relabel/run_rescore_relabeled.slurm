#!/bin/bash
#SBATCH -J lg_rescore_relabel
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Re-score the dolly paraphrases with the RELABELED Llama-Guard vector
# (layer 30, built from Llama-Guard's own safe/unsafe verdicts; AUC 0.974).
# Output goes to dolly/llamaguard_relabeled/ so the old (bad) llamaguard/ run
# stays intact for comparison. Scored in bf16 (--no_quantize) to MATCH the
# bf16 activations the vector was extracted from (avoids quant mismatch).

set -euo pipefail
echo "[stage] lg rescore-relabel start on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
fi
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
[ -n "${HUGGING_FACE_HUB_TOKEN}" ] || { echo "[error] no HF token"; exit 2; }

# read model from scratch cache, write nothing there (scratch can refill)
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export HF_DATASETS_CACHE="${HOME}/.cache/hf_datasets_relabel"
export TMPDIR="${HOME}/tmp_relabel"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
echo "[stage] python: $(python --version)"
nvidia-smi | head -12 || true

PROJECT_ROOT="${PROJECT_ROOT:-${HOME}/ClaRE}"
SCORE_PY="${PROJECT_ROOT}/research/overrefusal_finetuning/ppo_or/score_and_rank_all.py"
PARA_ROOT="${PROJECT_ROOT}/research/overrefusal_finetuning/paraphrases_api"
VEC="${PROJECT_ROOT}/research/refusal_vector/llamaguard_relabel/refusal_vector_llamaguard_relabeled.layer030.npz"

for f in "${SCORE_PY}" "${VEC}"; do [ -f "$f" ] || { echo "[error] missing $f"; exit 2; }; done
[ -f "${PARA_ROOT}/dolly/paraphrases.json" ] || { echo "[error] missing dolly paraphrases"; exit 2; }

cd "${PROJECT_ROOT}/research/overrefusal_finetuning/ppo_or"
python -u "${SCORE_PY}" \
  --paraphrases_root "${PARA_ROOT}" \
  --refusal_vector_path "${VEC}" \
  --scoring_model meta-llama/Llama-Guard-3-8B \
  --activation_layer 30 \
  --datasets dolly \
  --batch_size 32 \
  --no_quantize \
  --output_subdir llamaguard_fixed

echo "[done] rescore complete at $(date)"
ls -la "${PARA_ROOT}/dolly/llamaguard_relabeled/" 2>&1 || true
