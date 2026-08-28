#!/bin/bash
#SBATCH --job-name=rwr-icannot
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/rwr_icannot_%j.out
#SBATCH --error=logs/rwr_icannot_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Training RWR (icannot-OR reward) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Reward = exp(18.4*(sim-0.75)) * P("I cannot"|rewrite)  (d=1 -> reward IS icannot-OR)
# Absolute-threshold bins isolate the tiny high-signal tail (see build_icannot_shards.py).
python -u train_rwr.py \
    --shard_dir icannot_train_shards \
    --output_dir rwr_icannot_checkpoints \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --num_bins 5 \
    --bin_edges 1e-4,1e-3,1e-2,1e-1 \
    --bin_weights 1,8,24,64,128 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --seed 42

echo "Training complete at $(date)"
