#!/bin/bash
#SBATCH --job-name=rwr-llama-logit
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/rwr_llama_logit_%j.out
#SBATCH --error=logs/rwr_llama_logit_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# RWR Llama attacker on the multi-phrase refusal-onset logit reward: delta of the
# teacher-forced P(reply begins with any of the 5 mined Llama openers), k=18.4
# similarity gate. The old icannot line's single opener covers only ~57% of Llama
# refusals; the mined 5-opener set covers ~99.7%.
# Data: probe_or/results/llama_scored/llama_shards_logit (run_score_llama_or).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Absolute bin edges on the k=18.4 logit-OR, from the scoring job's log
# (logs/score_llama_or_*.out) -- use the build_llama_shards "candidate --bin_edges"
# line, computed over TRAINING-ONLY pairs (post-heldout). Data-dependent by design
# (the logit delta is heavily zero-inflated); set it after scoring; do not guess.
# SET 2026-08-13 from job 18993896. NOT the quantile edges: the logit-OR is extreme-tailed
# (p50 = 1.0e-07; only 17.5% of trainable pairs exceed 1e-4), so quantile bins put ~31% of
# sampling on pairs whose refusal-onset probability barely moved, and the q85 top bin has
# median OR only 0.013. Using the ABSOLUTE edges already validated on this signal family
# (logprob_or/experiment_brief.md §5) instead -- rwr_config.py documents exactly this case.
# Bin counts 18684/1271/882/775/1026 -> sampling 0/1.5/4.3/15.0/79.3%, i.e. 79% of training
# drawn from the 1,026 pairs with a real refusal-onset shift. Matches the icannot run's
# reported 1/4/15/79% shares almost exactly. NB: 5 bins here, so 4 edges + 5 weights.
BIN_EDGES="1e-4,1e-3,1e-2,1e-1"
[ -n "$BIN_EDGES" ] || { echo "ERROR: set BIN_EDGES in this wrapper from the score_llama_or log first"; exit 1; }

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Llama (multi-phrase logit reward, k=18.4) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u train_rwr.py \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --shard_dir probe_or/results/llama_scored/llama_shards_logit \
    --output_dir probe_or/results/rwr_llama_ckpt_logit \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 5 --bin_edges "$BIN_EDGES" --bin_weights 0,1,4,16,64 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
