#!/bin/bash
#SBATCH --job-name=rwr-qwen
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/rwr_qwen_%j.out
#SBATCH --error=logs/rwr_qwen_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Qwen (QLoRA) on Qwen-OR-scored Sonnet pool at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Reward = exp(18.4*(sim-0.75)) * probe_delta  (d=1). Training set after filter (sim>=0.5,
# OR>=0) = 1367 pairs / 714 originals. Bins on Qwen-OR: <0.1(462) .1-.5(399) .5-2(303) >2(203),
# weights [0,1,4,16] -> mass on the high-OR tail. num_samples 4000 keeps repetition ~13x.
python -u train_rwr.py \
    --base_model Qwen/Qwen3-32B --load_in_4bit \
    --shard_dir probe_or/results/qwen_shards \
    --output_dir probe_or/results/rwr_qwen_ckpt \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges 0.1,0.5,2 --bin_weights 0,1,4,16 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
