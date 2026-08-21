#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --job-name=delta-extract
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/delta_extract_%j.out
#SBATCH --error=logs/delta_extract_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u extract_delta_acts.py --sets probe_or/results/delta/prompt_sets.csv \
    --out_dir probe_or/results/delta --batch_size 16
