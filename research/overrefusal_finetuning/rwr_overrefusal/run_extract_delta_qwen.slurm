#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=delta-extract-qwen
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/delta_extract_qwen_%j.out
#SBATCH --error=logs/delta_extract_qwen_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u extract_delta_acts.py \
    --sets probe_or/results/delta_qwen/prompt_sets.csv \
    --out_dir probe_or/results/delta_qwen \
    --base_model Qwen/Qwen3-32B --batch_size 8
