#!/bin/bash
#SBATCH --job-name=rwr-llama-logit-tune
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/rwr_llama_logit_tune_%j.out
#SBATCH --error=logs/rwr_llama_logit_tune_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Tuning variants of the WINNING logit-reward Llama attacker (see
# LOGIT_TUNING_EXPERIMENTS.md). Identical to run_rwr_llama_logit.slurm except the
# three parameters exposed as arguments. Bin edges/weights, k, c, d unchanged —
# rwr_data.recompute_or_scores rebuilds OR from stored refusal_delta+similarity at
# load, so no rescoring job is needed for floor changes.
# Usage: sbatch run_rwr_llama_logit_tune.slurm <tag> <num_epochs> <sim_floor>
#   e.g.  sbatch run_rwr_llama_logit_tune.slurm _tune1_e6  6 0.5
#         sbatch run_rwr_llama_logit_tune.slurm _tune2_f70 3 0.7
set -euo pipefail
TAG="${1:?usage: sbatch run_rwr_llama_logit_tune.slurm <tag> <num_epochs> <sim_floor>}"
EPOCHS="${2:?num_epochs}"
FLOOR="${3:?sim_floor}"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Same validated absolute edges as run_rwr_llama_logit.slurm (job 18997726).
BIN_EDGES="1e-4,1e-3,1e-2,1e-1"

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Llama logit tune ${TAG}: epochs=${EPOCHS} sim_floor=${FLOOR} at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u train_rwr.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --shard_dir probe_or/results/llama_scored/llama_shards_logit \
    --output_dir "probe_or/results/rwr_llama_ckpt_logit${TAG}" \
    --num_epochs "$EPOCHS" --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 5 --bin_edges "$BIN_EDGES" --bin_weights 0,1,4,16,64 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor "$FLOOR" --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
