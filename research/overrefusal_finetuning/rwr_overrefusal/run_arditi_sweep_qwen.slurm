#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=qwen-arditi-sweep
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "### Arditi layer selection by ablation -- Qwen3-32B"
python -u arditi_layer_sweep.py --base_model Qwen/Qwen3-32B \
    --out_dir probe_or/results/arditi_sweep_qwen --batch_size 8

echo "### Llama mirror (same instrument, so the two are comparable)"
python -u arditi_layer_sweep.py --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --cand_layers 8,10,12,14,16,17,18,20,22,24,28,31 \
    --out_dir probe_or/results/arditi_sweep_llama --batch_size 16
