#!/bin/bash
#SBATCH --job-name=claude-score
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/claude_score_%j.out
#SBATCH --error=logs/claude_score_%j.err

# Score phase for raw-Claude held-out generations (generate first with
# generate_claude_heldout.py --skip_scoring on a node with internet+API key):
#   sbatch --export=ALL,OUTPUT_NAME=claude_heldout_alpaca.json run_score_claude_clusterb.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export HF_HOME="${HOME}/scratch/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PY=~/scratch/conda_envs/ppo-or-env/bin/python
OUTPUT_NAME="${OUTPUT_NAME:-claude_heldout_alpaca.json}"

echo "Starting claude score phase at $(date) (output_name=${OUTPUT_NAME})"
nvidia-smi --query-gpu=name --format=csv,noheader || true

$PY -u iterate_prompts_claude.py \
    --skip_generation \
    --output_name "${OUTPUT_NAME}" \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
    --scoring_model meta-llama/Meta-Llama-3-8B-Instruct \
    --score_batch_size 8

echo "claude score phase completed at $(date)"
