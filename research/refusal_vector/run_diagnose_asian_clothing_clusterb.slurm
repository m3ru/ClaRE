#!/bin/bash
#SBATCH -J asian_diag
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export DIAG_SCRATCH="${HOME}/scratch"
export HF_HOME="${DIAG_SCRATCH}/hf_cache" HF_HUB_CACHE="${DIAG_SCRATCH}/hf_cache/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DIAG_SCRATCH}/conda_envs/ppo-or-env"
cd "${HOME}/anon-repo/research/refusal_vector"
python -u diagnose_asian_clothing_refusal.py \
  > "${DIAG_SCRATCH}/prompt_opt/coherent_scale_v1/asian_clothing_diagnostic.json"
