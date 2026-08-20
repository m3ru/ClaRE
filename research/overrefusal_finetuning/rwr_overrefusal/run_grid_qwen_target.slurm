#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --job-name=grid-qwen-tgt
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/grid_%j.out
#SBATCH --error=logs/grid_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u score_target_refusal.py --rewrites_csv "$1" --output "$2" \
    --base_model Qwen/Qwen3-32B --n_samples 4 --max_new 64
