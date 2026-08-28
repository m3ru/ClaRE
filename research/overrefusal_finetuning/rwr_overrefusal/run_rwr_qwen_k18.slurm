#!/bin/bash
#SBATCH --job-name=rwr-qwen-k18
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/rwr_qwen_k18_%j.out
#SBATCH --error=logs/rwr_qwen_k18_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Qwen (QLoRA, k=18.4) on Qwen-OR-scored Sonnet pool at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# k=18.4 (primary): strong similarity gate -> minimal-edit over-refusals (top bin sim 0.844,
# delta 0.998). sim and probe_delta are ~independent (rho=-0.09), so the delta>=0 filter keeps
# top-OR examples refusal-inducing (59/60 delta>=0.4) despite the sim-dominated ranking.
# Bins on k=18.4 Qwen-OR: <0.1(462) .1-.5(399) .5-2(303) >2(203), weights [0,1,4,16].
python -u train_rwr.py \
    --base_model Qwen/Qwen3-32B --load_in_4bit \
    --shard_dir probe_or/results/qwen_shards \
    --output_dir probe_or/results/rwr_qwen_ckpt_k18 \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges 0.1,0.5,2 --bin_weights 0,1,4,16 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
