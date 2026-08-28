#!/bin/bash
#SBATCH -J lg_rescore_claude
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_HOME="${HOME}/scratch/hf_cache" HF_HUB_CACHE="${HOME}/scratch/hf_cache/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOME}/scratch/conda_envs/ppo-or-env"
D="${HOME}/ClaRE/research/refusal_vector/llamaguard_relabel"
cd "${D}"
python -u rescore_claude_dolly.py \
  --input "${D}/claude_dolly_rewrites.json" \
  --vector "${D}/refusal_vector_llamaguard_relabeled.layer030.npz" \
  --output "${D}/claude_dolly_rewrites_llamaguard.json" \
  --scoring_model meta-llama/Llama-Guard-3-8B --layer 30
