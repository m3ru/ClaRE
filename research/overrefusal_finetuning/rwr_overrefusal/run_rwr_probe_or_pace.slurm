#!/bin/bash
#SBATCH -J rwr_probe_or
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
echo "[stage] rwr_probe_or (top20 bins, probe-OR reward) on $(hostname) at $(date)"
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
python -u train_rwr.py \
  --shard_dir "${SCRATCH_BASE}/probe_or_shards" \
  --output_dir "${SCRATCH_BASE}/rwr_probe_or_checkpoints" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --num_epochs 3 --batch_size 4 --learning_rate 1.5e-5 --max_seq_length 512 \
  --num_bins 5 --bin_weights 0,0,0,0,1 \
  --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
  --similarity_floor 0.5 --seed 42 --gradient_checkpointing
echo "[done] rwr_probe_or at $(date)"; ls -la "${SCRATCH_BASE}/rwr_probe_or_checkpoints/final" 2>&1 | head
