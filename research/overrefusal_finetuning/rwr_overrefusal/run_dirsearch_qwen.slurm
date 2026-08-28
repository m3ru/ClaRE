#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=dirsearch-qwen
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=logs/dirsearch_qwen_%j.out
#SBATCH --error=logs/dirsearch_qwen_%j.err

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u search_selective_dirs.py --base_model Qwen/Qwen3-32B \
  --dirs_npz probe_or/results/delta_qwen/causal_dirs.npz \
  --atlas probe_or/results/qwen_signals/probe_absolute.npz \
  --n_random 12 --out probe_or/results/dirsearch_qwen_v2.json
