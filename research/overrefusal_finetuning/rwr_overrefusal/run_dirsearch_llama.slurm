#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=dirsearch-llama
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/dirsearch_llama_%j.out
#SBATCH --error=logs/dirsearch_llama_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u search_selective_dirs.py --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dirs_npz probe_or/results/delta/causal_dirs.npz \
  --atlas probe_or/results/llama_signals/probe_absolute.npz \
  --out probe_or/results/dirsearch_llama.json
