#!/bin/bash
#SBATCH -J abliteration
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
echo "[stage] abliteration on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"; export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
DIR="${HOME}/ClaRE/research/refusal_vector"; cd "${DIR}"
python -c "import torch,transformers,datasets" || { echo "[error] env"; exit 3; }
python -u abliteration_test.py \
  --direction_npz "${DIR}/arditi_direction_out/arditi_direction.npz" \
  --harmful_csv "${SCRATCH_BASE}/causal_refusal/harmful_behaviors.csv" \
  --out "${DIR}/abliteration_results.json" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --harmful_offset 300 --n_test 100 --max_new_tokens 48
echo "[done] abliteration at $(date)"
