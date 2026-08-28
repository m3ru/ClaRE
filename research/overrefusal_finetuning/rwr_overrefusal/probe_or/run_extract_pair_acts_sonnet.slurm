#!/bin/bash
#SBATCH --job-name=probe-pair-sonnet
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/probe_pair_sonnet_%j.out
#SBATCH --error=logs/probe_pair_sonnet_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs probe_or/pair_acts_sonnet

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Paired activation extraction (Sonnet pool, 12k pairs) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 12,000 pairs -> 24,000 forward passes; ~78/s on this MIG slice => ~10 min + model load.
python -u probe_or/extract_pair_acts.py \
    --pairs_csv probe_or/pairs_sonnet.csv \
    --out_dir probe_or/pair_acts_sonnet

echo "=== extraction done $(date); measuring delta probe inline (CPU) ==="
# Run the measurement here (not on the login node): 12k pairs need ~20GB transient RAM,
# which fits this job's 48G. Guarded with `|| true` so a probe-side hiccup can't mark the
# already-saved (expensive) extraction as FAILED -- rerun the probe on login if so.
python -u probe_or/probe_delta.py --acts_dir probe_or/activations \
    --pair_dir probe_or/pair_acts_sonnet --direction raw || echo "[warn] raw probe failed; rerun on login"
python -u probe_or/probe_delta.py --acts_dir probe_or/activations \
    --pair_dir probe_or/pair_acts_sonnet --direction lda || echo "[warn] lda probe failed; rerun on login"

echo "Done at $(date)"
