#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=qwen-safety-resp
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/qwen_safety_resp_%j.out
#SBATCH --error=logs/qwen_safety_resp_%j.err

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u gen_safety_responses.py --base_model Qwen/Qwen3-32B \
  --dirs_npz probe_or/results/delta_qwen/causal_dirs.npz \
  --atlas probe_or/results/qwen_signals/probe_absolute.npz \
  --d4_idx 5 --dir_label d6_exfiltration --batch_size 4 \
  --out probe_or/results/safety_responses_qwen.json
