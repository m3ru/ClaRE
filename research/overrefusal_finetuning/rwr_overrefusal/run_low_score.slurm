#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --job-name=low-score-llama
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/low_score_%j.out
#SBATCH --error=logs/low_score_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u score_target_refusal.py \
    --rewrites_csv probe_or/results/low_power/low_candidates.csv \
    --output      probe_or/results/low_power/low_scored_llamaTgt.json \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct --n_samples 4 --max_new 64
