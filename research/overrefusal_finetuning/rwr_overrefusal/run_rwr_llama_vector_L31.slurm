#!/bin/bash
#SBATCH --job-name=rwr-llama-vecL31
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/rwr_llama_vecL31_%j.out
#SBATCH --error=logs/rwr_llama_vecL31_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# RWR Llama attacker on the vector reward at the best-PREDICTOR layer L31 (Spearman 0.291
# vs causal L17's 0.261). Companion to run_rwr_llama_vector.slurm (L17); identical recipe,
# only the reward layer differs. Tests whether the vector reward's behavioral failure is
# about the layer or the signal type. NB: L31 is causally UNVALIDATED and adjacent to the
# inert L32 -- a win here is a Goodhart-risk finding, not a causal claim.
# Data: probe_or/results/llama_scored/llama_shards_vector_L31 (run_score_llama_or_L31).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Absolute bin edges on the k=18.4 vector-OR, from the scoring job's log
# (logs/score_llama_or_*.out) -- use the build_llama_shards "candidate --bin_edges"
# line, which is computed over TRAINING-ONLY pairs (post-heldout). Data-dependent:
# set it after the scoring job; do not guess.
# SET 2026-08-14 from job 19010437: q35/q65/q85 of L31-vector-OR over the 12,785 trainable
# training-only pairs. Verified well-spread (97% of trainable OR>0.01), so quantile binning
# is correct here (same as the L17 vector run) -- NOT the logit's extreme-tail case.
# Bin counts 4475/3835/2557/1918 -> sampling 0/8.6/22.9/68.6% (identical structure to L17
# vector). Top bin strong: min OR 8.14, median 17.3 (heavier tail than L17's median 3.98).
# 4 bins, weights [0,1,4,16] -- structurally identical to run_rwr_llama_vector.slurm; only
# the layer (31 vs 17) and its own OR quantiles differ. That is the comparison.
BIN_EDGES="0.3704,2.205,8.138"
[ -n "$BIN_EDGES" ] || { echo "ERROR: set BIN_EDGES from the score_llama_L31 build log first"; exit 1; }

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Llama (vector@L17 reward, k=18.4) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u train_rwr.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --shard_dir probe_or/results/llama_scored/llama_shards_vector_L31 \
    --output_dir probe_or/results/rwr_llama_ckpt_vector_L31 \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges "$BIN_EDGES" --bin_weights 0,1,4,16 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
