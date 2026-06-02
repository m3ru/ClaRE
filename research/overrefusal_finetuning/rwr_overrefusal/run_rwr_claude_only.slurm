#!/bin/bash
#SBATCH --job-name=rwr-claude
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/rwr_claude_train_%j.out
#SBATCH --error=logs/rwr_claude_train_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# RWR training on Claude-only deduped shards. Same hyperparameters that worked
# for RWR v3 (BinningConfig defaults: k=5.0, c=0.75, d=100, bin_weights=[0,0,0,1,16]
# i.e. top-40% only).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting RWR Claude-only training at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u train_rwr.py \
    --shard_dir ./prompt_iteration_results/dataset_research_framing_full_shards_deduped \
    --output_dir ./rwr_claude_checkpoints \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 5 \
    --similarity_floor 0.5 \
    --seed 42 \
    --gradient_checkpointing

echo "RWR Claude-only training completed at $(date)"
