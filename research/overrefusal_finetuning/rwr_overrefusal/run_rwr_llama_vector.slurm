#!/bin/bash
#SBATCH --job-name=rwr-llama-vector
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/rwr_llama_vector_%j.out
#SBATCH --error=logs/rwr_llama_vector_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# RWR Llama attacker on the CORRECTED vector reward: delta projection at the causal
# layer L17 (ablation 99->83, addition 0->99), k=18.4 similarity gate (Qwen-validated
# recipe). Data: probe_or/results/llama_scored/llama_shards_vector (run_score_llama_or).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Absolute bin edges on the k=18.4 vector-OR, from the scoring job's log
# (logs/score_llama_or_*.out) -- use the build_llama_shards "candidate --bin_edges"
# line, which is computed over TRAINING-ONLY pairs (post-heldout). Data-dependent:
# set it after the scoring job; do not guess.
# SET 2026-08-13 from job 18993896: q35/q65/q85 of vector-OR over the 18,683 trainable
# training-only pairs. Bin counts 6539/5605/3736/2803 -> sampling 0/8.6/22.9/68.6%.
# Top bin is genuinely strong (min OR 2.04, median 3.98). Near-identical to the validated
# Qwen k18 edges (0.1,0.5,2) -- convergent evidence the scale is right.
BIN_EDGES="0.09933,0.5765,2.038"
[ -n "$BIN_EDGES" ] || { echo "ERROR: set BIN_EDGES in this wrapper from the score_llama_or log first"; exit 1; }

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Llama (vector@L17 reward, k=18.4) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u train_rwr.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --shard_dir probe_or/results/llama_scored/llama_shards_vector \
    --output_dir probe_or/results/rwr_llama_ckpt_vector \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges "$BIN_EDGES" --bin_weights 0,1,4,16 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
