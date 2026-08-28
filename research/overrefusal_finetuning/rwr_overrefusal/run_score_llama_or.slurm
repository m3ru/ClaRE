#!/bin/bash
#SBATCH --job-name=score-llama-or
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/score_llama_or_%j.out
#SBATCH --error=logs/score_llama_or_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Score the harm-filtered Sonnet benign pool with the three CORRECTED Llama refusal
# signals (vector@L17 causal / delta-probe ensemble / multi-phrase opener logit),
# then build the three RWR shard sets + the common held-out split.
# Replaces the retired layer-32 vector OR score. Pick --bin_edges for the training
# wrappers from this job's log (per-signal "suggested --bin_edges" lines).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/llama_scored

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Llama 3-signal pool scoring at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u probe_or/score_llama_or.py \
    --pairs_csv logprob_or/results/icannot_or_sonnet/icannot_or_pairs_benign.csv \
    --scorer probe_or/results/llama_signals/qwen_probe_raw.npz \
    --vector_layer 17 \
    --opener_json refusal_atlas/opener_sets.json --model_key llama \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --out probe_or/results/llama_scored/sonnet_benign_llama_or.csv \
    --k 18.4 --c 0.75 --proj_bs 16 --logit_bs 16

python -u probe_or/build_llama_shards.py \
    --scored_csv probe_or/results/llama_scored/sonnet_benign_llama_or.csv \
    --out_root probe_or/results/llama_scored \
    --heldout_fraction 0.1 --seed 42 --n_shards 40

echo "Done at $(date)"
