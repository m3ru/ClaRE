#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=safety-resp
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/safety_resp_%j.out
#SBATCH --error=logs/safety_resp_%j.err

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u gen_safety_responses.py
