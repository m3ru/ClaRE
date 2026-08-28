#!/bin/bash
#SBATCH --job-name=atlas-scaleup-llama
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/atlas_scaleup_llama_%A_%a.out
#SBATCH --error=logs/atlas_scaleup_llama_%A_%a.err
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# 3-signal (vector@L17 / probe / logit-openers) + behavioral scoring of the logit
# attacker's scale-up rewrites. Sharded by pair; concat the per-prompt/pairs CSVs after.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/scaleup_atlas_llama

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

S=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
echo "atlas scale-up shard ${S} at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u refusal_atlas/score_signals.py \
    --substrate probe_or/results/scaleup_atlas_llama/substrate_scaleup_shard${S}.csv \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --probe_npz probe_or/results/llama_signals/probe_absolute.npz --best_layer 17 \
    --opener_json refusal_atlas/opener_sets.json --model_key llama \
    --out probe_or/results/scaleup_atlas_llama/signals_scaleup_shard${S}.csv \
    --pairs_out probe_or/results/scaleup_atlas_llama/pairs_scaleup_shard${S}.csv \
    --samples_out probe_or/results/scaleup_atlas_llama/samples_scaleup_shard${S}.json \
    --n_samples 4 --temp 0.7 --max_new 96 \
    --proj_bs 16 --logit_bs 16 --gen_bs 16 --max_length 512 --seed 42

echo "done shard ${S} at $(date)"
