#!/bin/bash
#SBATCH -J arditi_dir
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
echo "[stage] arditi_dir on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"; export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
DIR="${HOME}/anon-repo/research/refusal_vector"; cd "${DIR}"
python -c "import torch,transformers,datasets" || { echo "[error] env"; exit 3; }
python -u arditi_direction_test.py \
  --harmful_csv "${SCRATCH_BASE}/causal_refusal/harmful_behaviors.csv" \
  --out_dir "${DIR}/arditi_direction_out" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --n_dir 256 --n_val 40 --n_test 100 \
  --cand_layers 8,10,12,14,16,18,20,22,24,28 --add_coefs 1,2,4 --max_new_tokens 40
echo "[done] arditi_dir at $(date)"
