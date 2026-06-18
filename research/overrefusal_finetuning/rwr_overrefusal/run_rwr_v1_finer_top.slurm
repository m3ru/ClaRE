#!/bin/bash
#SBATCH --job-name=rwr-claude-v1
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/rwr_claude_v1_train_%j.out
#SBATCH --error=logs/rwr_claude_v1_train_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# V1: finer top bins. Same total dropped fraction (70%) as the original
# claude_rwr (which used num_bins=5 + bin_weights=[0,0,0,1,16]), but split the
# kept 30% into three deciles so the very top can be weighted more aggressively.
#
#     --num_bins 10  --bin_weights 0,0,0,0,0,0,0,1,8,32
#
# Motivation (from analyze_bucketing.py output on the Claude shards):
#   bin 7 of 10 (top 30-20%): OR mean 0.0794
#   bin 8 of 10 (top 20-10%): OR mean 0.1666
#   bin 9 of 10 (top 10%):    OR mean 0.2585  <- 1.55x bin 8, 3.26x bin 7
#
# The current weight-16 lumps the top 20% together; this concentrates training
# mass on the top decile while still using the next two deciles as supporting
# signal. All other hyperparameters identical to `run_rwr_claude_only.slurm`
# so the eval delta is attributable to the binning change alone.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting RWR Claude V1 (finer top bins) training at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u train_rwr.py \
    --shard_dir ./prompt_iteration_results/dataset_research_framing_full_shards_deduped \
    --output_dir ./rwr_claude_v1_checkpoints \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 10 \
    --bin_weights 0,0,0,0,0,0,0,1,8,32 \
    --similarity_floor 0.5 \
    --seed 42 \
    --gradient_checkpointing

echo "RWR Claude V1 training completed at $(date)"
