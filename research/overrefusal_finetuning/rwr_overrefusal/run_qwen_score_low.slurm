#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=qwen-low-score
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/qwen_low_score_%j.out
#SBATCH --error=logs/qwen_low_score_%j.err

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u score_target_refusal.py \
    --rewrites_csv probe_or/results/low_power_qwen/low_candidates.csv \
    --output      probe_or/results/low_power_qwen/low_scored_qwenTgt.json \
    --base_model Qwen/Qwen3-32B --n_samples 4 --max_new 64
