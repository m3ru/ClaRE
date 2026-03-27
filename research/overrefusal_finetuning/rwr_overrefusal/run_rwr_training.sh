#!/bin/bash
#SBATCH --job-name=rwr-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --output=logs/rwr_train_%j.out
#SBATCH --error=logs/rwr_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mgopalan6@gatech.edu

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/.bashrc 2>/dev/null || true
conda activate CHANGE_ME

set -eo pipefail

echo "Starting RWR training at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python train_rwr.py \
    --shard_dir ../or_paraphrase_3k \
    --output_dir ./rwr_checkpoints \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 5 \
    --similarity_floor 0.5 \
    --seed 42 \
    --gradient_checkpointing

echo "RWR training completed at $(date)"
