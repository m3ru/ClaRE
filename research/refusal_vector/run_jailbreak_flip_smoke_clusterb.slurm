#!/bin/bash
#SBATCH -J jb_smoke
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export JB_SCRATCH="${HOME}/scratch"
export HF_HOME="${JB_SCRATCH}/hf_cache" HF_HUB_CACHE="${JB_SCRATCH}/hf_cache/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${JB_SCRATCH}/conda_envs/ppo-or-env"
cd "${HOME}/anon-repo/research/refusal_vector"
OUT="${JB_SCRATCH}/prompt_opt/jailbreak_flip_v1/smoke"
DATA="${JB_SCRATCH}/prompt_opt/jailbreak_flip_v1/data/strongreject_custom_direct.csv"
mkdir -p "${OUT}"
python -u prompt_rewrite_opt.py \
  --prompts_csv "${DATA}" --prompt_column prompt \
  --dirs_npz "${JB_SCRATCH}/prompt_opt/harmful_dirs_all_layers.npz" \
  --out "${OUT}/anti_smoke.json" --objective anti_icannot --layer 12 \
  --n_prompts 2 --max_prompt_tok 96 --steps 50 --topk 64 --n_cand 32 \
  --sim_floor 0.85 --lam 20 --fluency_weight 1 --lm_topk_filter 64 \
  --editable_single_token_words_only --max_edits 4 \
  --max_nll_increase 0.15 --max_nll_max_increase 1.0 --max_new_tokens 64
