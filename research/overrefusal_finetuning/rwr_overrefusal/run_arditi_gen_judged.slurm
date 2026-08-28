#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=arditi-gen-judged
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/arditi_gen_%j.out
#SBATCH --error=logs/arditi_gen_%j.err
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
# Phase 1 of the judged Arditi sweep: generate only, classify nothing. Both models, so the
# Llama selection is re-derived on the same instrument and the comparison is like-for-like.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import torch, transformers; print('[pre-flight] imports OK')"

echo "### Qwen3-32B  $(date)"
python -u arditi_sweep_generate.py --base_model Qwen/Qwen3-32B \
    --cand_layers 16,20,24,28,32,36,40,44,48,52,56,57,60 \
    --out_dir probe_or/results/arditi_judged_qwen --batch_size 8

echo "### Llama-3-8B mirror  $(date)"
python -u arditi_sweep_generate.py --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --cand_layers 8,10,12,14,16,17,18,20,22,24,28,31 \
    --out_dir probe_or/results/arditi_judged_llama --batch_size 16
echo "### done $(date)"
