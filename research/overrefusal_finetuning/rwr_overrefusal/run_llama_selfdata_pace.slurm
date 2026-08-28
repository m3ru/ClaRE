#!/bin/bash
#SBATCH --job-name=llama-selfdata
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/llama_selfdata_%j.out
#SBATCH --error=logs/llama_selfdata_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export HF_HOME="${HOME}/scratch/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_DATASETS_OFFLINE=1
PY=~/scratch/conda_envs/ppo-or-env/bin/python
ARGS="${ARGS:---n_alpaca 2500 --n_dolly 2500 --n_per_prompt 8 --temperature 1.0 --out_dir ../llama_selfparaphrase_5k}"
echo "start $(date); ARGS=$ARGS"; nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -u generate_llama_selfdata.py $ARGS
echo "done $(date)"
