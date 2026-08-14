#!/bin/bash
#SBATCH --job-name=score-llama-L31
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/score_llama_L31_%j.out
#SBATCH --error=logs/score_llama_L31_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Re-score the SAME pool with the vector delta taken at L31 (the best-PREDICTOR layer,
# Spearman 0.291) instead of the causal L17 (0.261), to test whether the vector reward's
# behavioral failure is about the LAYER or about the SIGNAL TYPE. IDENTICAL code path to
# run_score_llama_or.slurm -- only --vector_layer differs -- so probe_delta and logit are
# recomputed byte-identically (a built-in soundness check, verified below). Then build
# ONLY the vector shard set, tagged _L31, REUSING the L17 held-out split so the eval
# prompts are identical and the two vector attackers are directly comparable.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/llama_scored

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Llama vector re-score at L31 at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u probe_or/score_llama_or.py \
    --pairs_csv logprob_or/results/icannot_or_sonnet/icannot_or_pairs_benign.csv \
    --scorer probe_or/results/llama_signals/qwen_probe_raw.npz \
    --vector_layer 31 \
    --opener_json refusal_atlas/opener_sets.json --model_key llama \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --out probe_or/results/llama_scored/sonnet_benign_llama_or_L31.csv \
    --k 18.4 --c 0.75 --proj_bs 16 --logit_bs 16

# SOUNDNESS CHECK: probe_delta and logit columns must equal the L17 run exactly
# (only the vector layer changed). Aborts the whole job if they drift.
python - <<'PY'
import csv, sys
import numpy as np
csv.field_size_limit(sys.maxsize)
def load(p):
    return {(r["original"], r["rewrite"]): r for r in csv.DictReader(open(p))}
a = load("probe_or/results/llama_scored/sonnet_benign_llama_or.csv")       # L17
b = load("probe_or/results/llama_scored/sonnet_benign_llama_or_L31.csv")   # L31
assert a.keys() == b.keys(), f"row-set mismatch L17={len(a)} L31={len(b)}"
# Only the recomputed PRIMITIVES matter (or_* are deterministic fns of d_*+sim; sim is copied
# from the shared input CSV). Use a RELATIVE tolerance: bf16 matmuls aren't bit-identical across
# GPUs and %.6g values reach ~159 (ulp ~1.6e-3), so an absolute 1e-6 would false-abort. rtol 1e-3
# still catches any REAL divergence (wrong openers/scorer/context move these by >>0.1%).
bad = 0
for k in a:
    if not (np.isclose(float(a[k]["d_probe"]), float(b[k]["d_probe"]), rtol=1e-3, atol=1e-4)
            and np.isclose(float(a[k]["d_logit"]), float(b[k]["d_logit"]), rtol=1e-3, atol=1e-4)):
        bad += 1
if bad:
    raise SystemExit(f"[abort] {bad}/{len(a)} rows: probe/logit differ >0.1% between L17 and L31 "
                     "-- the re-score is NOT a clean layer swap; do not train on it.")
print(f"[soundness] probe+logit agree within fp tol across L17/L31 for all {len(a)} rows -- clean vector-only swap")
PY

# Build ONLY the vector shard set, tagged _L31, reusing the L17 held-out split.
python -u probe_or/build_llama_shards.py \
    --scored_csv probe_or/results/llama_scored/sonnet_benign_llama_or_L31.csv \
    --out_root probe_or/results/llama_scored \
    --signals vector --tag _L31 \
    --heldout_json probe_or/results/llama_scored/heldout_originals.json \
    --seed 42 --n_shards 40

echo "Done at $(date)"
