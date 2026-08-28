#!/bin/bash
#SBATCH --job-name=arditi-extract
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/arditi_extract_%j.out
#SBATCH --error=logs/arditi_extract_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Per-layer last-token activations for the ARDITI-FAITHFUL class split:
# HARMFUL (AdvBench, 512) vs HARMLESS (alpaca-cleaned, 512, disjoint from every
# eval-pair text). Feeds compare_directions.py, which asks which direction+layer
# best ranks behaviorally-induced over-refusal on benign rewrites:
#   ours   = mass-mean(refused - complied) on the Reddit/jailbreak split  -> L17
#   arditi = mass-mean(harmful - harmless)                                -> L12 (<collaborator>'s)
# Reuses the SAME validated extractor as the ours-split acts (right-pad, last real
# token, system="You are a helpful assistant.", add_generation_prompt) so the two
# direction sets live in an identical activation space and are comparable.
# float32: Llama is fp16-safe here, but keeps the LDA covariance solve well-conditioned.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir (parent of probe_or/)
mkdir -p logs probe_or/activations_arditi

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Extracting Arditi-split activations at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u probe_or/extract_layer_acts.py \
    --refusal_csv probe_or/data/arditi_harmful.csv \
    --benign_csv  probe_or/data/arditi_harmless.csv \
    --out_dir probe_or/activations_arditi \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_per_class 512 --batch_size 16 --acts_dtype float32

echo "Extraction complete at $(date)"
