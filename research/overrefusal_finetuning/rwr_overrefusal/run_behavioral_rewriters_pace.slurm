#!/bin/bash
#SBATCH -J behav_rewriters
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Behavioral refusal eval: generate 4 rewrites/prompt from baseline + top20 on
# held-out alpaca, feed each to bare Llama-3-8B, regex-count real refusals.
set -euo pipefail
echo "[stage] behav_rewriters on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
python -c "import torch,transformers,peft" || { echo "[error] env broken"; exit 3; }

RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"
cd "${RWR_DIR}"
TOP20="${SCRATCH_BASE}/rwr_binsweep/top20_only/final"
[ -d "${TOP20}" ] || { echo "[error] missing ${TOP20}"; exit 2; }

python -u behavioral_rewriters_eval.py \
  --top20_adapter "${TOP20}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --out_dir prompt_iteration_results/behavioral_rewriters \
  --n_prompts 200 \
  --n_per_prompt 4 \
  --rewrite_max_new_tokens 64 \
  --rewrite_temperature 0.7 \
  --resp_max_new_tokens 256 \
  --resp_batch 16 \
  --eval_seed 99 --training_seed 42 --training_n_alpaca 2500 --iter_seed 42 --iter_n 20

echo "[done] behav_rewriters complete at $(date)"
