#!/bin/bash
#SBATCH --job-name=atlas-sig-llama
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/atlas_sig_llama_%j.out
#SBATCH --error=logs/atlas_sig_llama_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs refusal_atlas/results

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Refusal Atlas signal+behavior scoring (Llama-3-8B) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Llama vector = L17: <collaborator>'s Arditi causal test (research/refusal_vector/causal_refusal_results.json)
# shows ablating L17 drops refusal 99->83% and adding it induces 0->99%, while L32 does neither.
# --probe_npz is refit (Qwen-style mass-mean + broad behavioral label); --best_layer pins the vector.
python -u refusal_atlas/score_signals.py \
    --substrate refusal_atlas/data/substrate.csv \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --probe_npz probe_or/results/llama_signals/probe_absolute.npz --best_layer 17 \
    --opener_json refusal_atlas/opener_sets.json --model_key llama \
    --out refusal_atlas/results/signals_llama.csv \
    --pairs_out refusal_atlas/results/signals_llama_pairs.csv \
    --samples_out refusal_atlas/results/samples_llama.json \
    --n_samples 8 --temp 0.7 --max_new 128 \
    --proj_bs 16 --logit_bs 16 --gen_bs 12 --max_length 512 --seed 42

echo "Done at $(date)"
