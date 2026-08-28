#!/bin/bash
#SBATCH --job-name=llama-signals-lda
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/llama_signals_lda_%j.out
#SBATCH --error=logs/llama_signals_lda_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

# CPU-only. Fits the Llama delta VECTOR + PROBE with COVARIANCE-CORRECTED (LDA)
# mass-mean directions -- Sigma^-1 (mu_ref - mu_ben), the Marks & Tegmark variant
# that is Bayes-optimal under shared-covariance Gaussians. Llama only ever got the
# `raw` fit (results/llama_signals/qwen_probe_raw.npz); Qwen has both, and there LDA
# LOSES badly (Spearman 0.646 vs 0.809) -- consistent with the covariance being
# estimated from ~2500 samples in 4096-5120 dims, where estimation error eats the
# theoretical optimum. This produces the DEPLOYABLE lda scorer so the raw-vs-LDA
# choice can be made on Llama's own numbers; writes qwen_{vector,probe}_lda.npz +
# qwen_signals_summary_lda.json (distinct names -- does not clobber the raw fit).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/llama_signals

module load scipy-stack/2024b          # numpy/scipy; sklearn from ~general
source ~/general/bin/activate

echo "Llama LDA signal fit at $(date)"

python -u probe_or/probe_qwen_signals.py \
    --acts_dir probe_or/activations \
    --pair_dir probe_or/pair_acts \
    --behav_csv probe_or/results/llama_behav/behav.csv \
    --out_dir probe_or/results/llama_signals \
    --direction lda --dP_thresh 0.01

echo "Done at $(date)"
