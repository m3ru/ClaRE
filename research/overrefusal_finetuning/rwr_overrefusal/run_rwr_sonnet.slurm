#!/bin/bash
#SBATCH --job-name=rwr-sonnet
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/rwr_sonnet_%j.out
#SBATCH --error=logs/rwr_sonnet_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs

module load cuda scipy-stack/2024b          # no arrow needed for training
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Training RWR on Sonnet pool (aggressive upweighting) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Reward = exp(18.4*(sim-0.75)) * P("I cannot")  (d=1). bin 0 (near-zero OR) zeroed;
# mass concentrated on OR>0.01 (bins 3-4, ~1850 pairs / 600+ sources). num_samples
# capped at 10k so the ~930 bin-4 examples aren't over-repeated. Checkpoints per epoch.
python -u train_rwr.py \
    --shard_dir logprob_or/results/shards_sonnet \
    --output_dir logprob_or/results/rwr_sonnet_ckpt \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --num_bins 5 --bin_edges 1e-4,1e-3,1e-2,1e-1 --bin_weights 0,1,4,16,64 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 10000 --seed 42

echo "Training complete at $(date)"
