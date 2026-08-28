#!/bin/bash
#SBATCH -J causal_refusal
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

# Arditi-style causal validation of our refusal direction (ablation + addition),
# testing the direction taken from layer 17 vs 32, on AdvBench (harmful) + benign.
set -euo pipefail
echo "[stage] causal_refusal on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
python -c "import torch,transformers" || { echo "[error] env"; exit 3; }

DIR="${HOME}/anon-repo/research/refusal_vector"
cd "${DIR}"
python -u causal_refusal_test.py \
  --harmful_csv "${SCRATCH_BASE}/causal_refusal/harmful_behaviors.csv" \
  --benign_csv  "${SCRATCH_BASE}/split_samples/split_benign_sample.csv" \
  --refusal_csv "${SCRATCH_BASE}/split_samples/split_refusal_sample.csv" \
  --out "${DIR}/causal_refusal_results.json" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --n_harmful 100 --n_harmless 100 --n_dir 2000 \
  --layers 17,32 --add_coefs 1,2,4 --max_new_tokens 48 --batch_size 16
echo "[done] causal_refusal at $(date)"
