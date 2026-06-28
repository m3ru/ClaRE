#!/bin/bash
#SBATCH -J score_analyze
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Score all generated alpaca OR adaptations with the Llama-3 OR score (fixed
# scorer), take top 10%, analyze most common words. Auto-runs after the
# generation array finishes (chained afterany; loads whatever shards exist).
set -euo pipefail
echo "[stage] score_analyze on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"

RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"
VEC="${HOME}/ClaRE/research/refusal_vector/Vector_Extraction/refusal_vector.layer032.npz"
GEN_DIR="${RWR_DIR}/gen_alpaca_or"
cd "${RWR_DIR}"
echo "[stage] gen shards present: $(ls ${GEN_DIR}/gen_alpaca_or_shard*.json 2>/dev/null | wc -l)"

python -u score_and_analyze.py \
  --gen_dir "${GEN_DIR}" \
  --vector "${VEC}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --activation_layer 32 \
  --k 5.0 --c 0.75 --d 100.0 \
  --top_frac 0.10 \
  --output "${GEN_DIR}/score_analysis_results.json"

echo "[done] score_analyze complete at $(date)"
