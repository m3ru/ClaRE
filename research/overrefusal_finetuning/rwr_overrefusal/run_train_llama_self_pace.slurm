#!/bin/bash
#SBATCH --job-name=llama-self-train
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/llama_self_train_%j.out
#SBATCH --error=logs/llama_self_train_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export HF_HOME="${HOME}/scratch/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_DATASETS_OFFLINE=1
PY=~/scratch/conda_envs/ppo-or-env/bin/python
echo "start $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -u train_rwr.py \
  --shard_dir ../llama_selfparaphrase_5k \
  --output_dir ~/scratch/rwr_llama_self_checkpoints \
  --bin_weights 0,0,0,1,16 \
  --similarity_exponent 5.0 --similarity_center 0.75 --refusal_divisor 100.0 \
  --num_epochs 3
echo "done $(date)"
