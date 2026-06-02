#!/bin/bash
#SBATCH --job-name=rwr-orp3k
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/rwr_orp3k_train_%j.out
#SBATCH --error=logs/rwr_orp3k_train_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Baseline RWR training on or_paraphrase_3k_deduped (gpt-oss-120b blind paraphrases).
# Same hyperparameters as the Claude-only run for a fair A/B comparison.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting RWR orp3k baseline training at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u train_rwr.py \
    --shard_dir ../or_paraphrase_3k_deduped \
    --output_dir ./rwr_orp3k_baseline_checkpoints \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 5 \
    --similarity_floor 0.5 \
    --seed 42 \
    --gradient_checkpointing

echo "RWR orp3k baseline training completed at $(date)"
