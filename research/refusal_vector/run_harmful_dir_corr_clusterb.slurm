#!/bin/bash
#SBATCH -J harmful_dir_corr
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
echo "[stage] harmful_dir_corr on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"; export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
DIR="${HOME}/anon-repo/research/refusal_vector"; cd "${DIR}"
python -c "import torch,transformers,scipy,datasets" || { echo "[error] env"; exit 3; }
python -u harmful_dir_vs_logit_corr.py \
  --harmful_csv "${SCRATCH_BASE}/causal_refusal/harmful_behaviors.csv" \
  --pairs_csv ${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal/probe_or/pairs.csv ${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal/probe_or/pairs_sonnet.csv \
  --out "${DIR}/harmful_dir_vs_logit_results.json" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct --n_dir 256 --sim_hi 0.85
echo "[done] harmful_dir_corr at $(date)"
