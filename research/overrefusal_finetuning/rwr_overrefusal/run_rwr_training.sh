#!/bin/bash
#SBATCH --job-name=rwr-train
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/rwr_train_%j.out
#SBATCH --error=logs/rwr_train_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting RWR training at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -u train_rwr.py \
    --shard_dir ../or_paraphrase_3k,./scored_taskaware \
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
