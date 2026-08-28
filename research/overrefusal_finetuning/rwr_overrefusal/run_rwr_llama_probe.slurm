#!/bin/bash
#SBATCH --job-name=rwr-llama-probe
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/rwr_llama_probe_%j.out
#SBATCH --error=logs/rwr_llama_probe_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# RWR Llama attacker on the delta-probe-ensemble reward (Aug-10 refit on broad
# behavioral labels), k=18.4 similarity gate (Qwen-validated recipe).
# Data: probe_or/results/llama_scored/llama_shards_probe (run_score_llama_or).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Absolute bin edges on the k=18.4 probe-OR, from the scoring job's log
# (logs/score_llama_or_*.out) -- use the build_llama_shards "candidate --bin_edges"
# line, which is computed over TRAINING-ONLY pairs (post-heldout). Data-dependent:
# set it after the scoring job; do not guess.
# SET 2026-08-13 from job 18993896: q35/q65/q85 of probe-OR over the 9,512 trainable
# training-only pairs. Bin counts 3329/2854/1902/1427 -> sampling 0/8.6/22.9/68.6%.
# Top bin min OR 2.45, median 5.53. Smaller pool than the other arms because the probe
# calls only 32.5% of rewrites refusal-increasing (vs vector 61.7%, logit 74.8%).
BIN_EDGES="0.1004,0.6119,2.447"
[ -n "$BIN_EDGES" ] || { echo "ERROR: set BIN_EDGES in this wrapper from the score_llama_or log first"; exit 1; }

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Llama (probe-ensemble reward, k=18.4) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u train_rwr.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --shard_dir probe_or/results/llama_scored/llama_shards_probe \
    --output_dir probe_or/results/rwr_llama_ckpt_probe \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges "$BIN_EDGES" --bin_weights 0,1,4,16 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
