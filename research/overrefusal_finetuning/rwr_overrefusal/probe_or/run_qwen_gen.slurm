#!/bin/bash
#SBATCH --job-name=qwen-gen
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/qwen_gen_%j.out
#SBATCH --error=logs/qwen_gen_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

# Parameterized via env vars (pass with: sbatch --export=ALL,PAIRS=...,OUT=...,LIMIT=... run_qwen_gen.slurm)
#   PAIRS : pairs CSV (default probe_or/pairs_qwen.csv)
#   OUT   : output CSV (default probe_or/results/qwen_behav/behav.csv)
#   LIMIT : first-N pairs (default 0 = all; use 200 for calibration)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PAIRS="${PAIRS:-probe_or/pairs_qwen.csv}"
OUT="${OUT:-probe_or/results/qwen_behav/behav.csv}"
LIMIT="${LIMIT:-0}"
mkdir -p logs "$(dirname "$OUT")"

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Qwen3-32B behavioral generation at $(date)  PAIRS=$PAIRS OUT=$OUT LIMIT=$LIMIT"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u probe_or/gen_qwen_refusal.py \
    --pairs_csv "$PAIRS" --out "$OUT" \
    --base_model Qwen/Qwen3-32B \
    --n_samples 4 --temperature 0.7 --max_new_tokens 64 --batch_size 8 --limit "$LIMIT"

echo "Done at $(date)"
