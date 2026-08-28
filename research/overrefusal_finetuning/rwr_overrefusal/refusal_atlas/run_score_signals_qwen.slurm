#!/bin/bash
#SBATCH --job-name=atlas-sig-qwen
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/atlas_sig_qwen_%j.out
#SBATCH --error=logs/atlas_sig_qwen_%j.err
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

echo "Refusal Atlas signal+behavior scoring (Qwen3-32B) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Qwen vector = L58 (causally validated: strongest addition-sufficiency 89% + strong ablation).
# --probe_npz = absolute-space refit; the probe collapses to L58, so probe ~ vector for Qwen.
# Full 80GB card required (32B bf16 ~65GB) -- NOT the MIG slice.
python -u refusal_atlas/score_signals.py \
    --substrate refusal_atlas/data/substrate.csv \
    --base_model Qwen/Qwen3-32B \
    --probe_npz probe_or/results/qwen_signals/probe_absolute.npz --best_layer 58 \
    --opener_json refusal_atlas/opener_sets.json --model_key qwen \
    --out refusal_atlas/results/signals_qwen.csv \
    --pairs_out refusal_atlas/results/signals_qwen_pairs.csv \
    --samples_out refusal_atlas/results/samples_qwen.json \
    --n_samples 8 --temp 0.7 --max_new 128 \
    --proj_bs 8 --logit_bs 8 --gen_bs 4 --max_length 512 --seed 42

echo "Done at $(date)"
