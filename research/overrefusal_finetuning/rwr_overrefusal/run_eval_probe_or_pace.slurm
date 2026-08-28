#!/bin/bash
#SBATCH -J eval_probe_or
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
echo "[stage] eval_probe_or (behavioral) on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"; cd "${RWR_DIR}"
python -c "import torch,transformers,peft" || { echo "[error] env"; exit 3; }
python -u behavioral_rewriters_eval.py \
  --top20_adapter "${SCRATCH_BASE}/rwr_probe_or_checkpoints/final" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --out_dir prompt_iteration_results/behavioral_probe_or \
  --n_prompts 200 --n_per_prompt 4 --resp_max_new_tokens 256 --resp_batch 16 \
  --eval_seed 99 --training_seed 42 --training_n_alpaca 2500 --iter_seed 42 --iter_n 20
echo "[done] eval_probe_or at $(date)"
