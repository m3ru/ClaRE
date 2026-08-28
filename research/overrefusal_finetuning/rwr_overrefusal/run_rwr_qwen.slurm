#!/bin/bash
#SBATCH --job-name=rwr-qwen
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/rwr_qwen_%j.out
#SBATCH --error=logs/rwr_qwen_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Qwen (QLoRA) on Qwen-OR-scored Sonnet pool at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Reward = exp(5.0*(sim-0.75)) * probe_delta  (d=1). k=5.0 (not the logprob runs' 18.4):
# probe_delta is a standardized z-score, so at k=18.4 the exp(sim) term dominates the ranking
# (rho(OR,sim)=0.85 vs rho(OR,delta)=0.39). At k=5.0 the refusal signal drives it
# (rho(OR,delta)=0.85, rho(OR,sim)=0.40) with similarity as a gate.
# Training set after filter (sim>=0.5, OR>=0) = 1367 pairs / 714 originals. Bins on k=5 Qwen-OR:
# <0.15(241) .15-.5(447) .5-1.5(561) >1.5(118), delta-means 0.12/0.50/1.06/1.51.
python -u train_rwr.py \
    --base_model Qwen/Qwen3-32B --load_in_4bit \
    --shard_dir probe_or/results/qwen_shards \
    --output_dir probe_or/results/rwr_qwen_ckpt \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges 0.15,0.5,1.5 --bin_weights 0,1,4,16 \
    --similarity_exponent 5.0 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
