#!/bin/bash
#SBATCH --job-name=cmp-directions
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/cmp_directions_%j.out
#SBATCH --error=logs/cmp_directions_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

# CPU-only. Decides two things from one labeled set (6000 pairs, behavioral dP):
#   Q1 which direction+layer -> ours(refused-vs-complied)@L17 vs arditi(harmful-vs-harmless)@L12
#   Q2 raw mass-mean vs covariance-corrected (LDA) -- the probe variant to deploy
# Needs the arditi extraction (run_extract_arditi_acts.slurm) to have finished.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results

module load scipy-stack/2024b
source ~/general/bin/activate

echo "Direction comparison at $(date)"
python -u probe_or/compare_directions.py \
    --ours_acts probe_or/activations \
    --arditi_acts probe_or/activations_arditi \
    --pair_dir probe_or/pair_acts \
    --behav_csv probe_or/results/llama_behav/behav.csv \
    --out probe_or/results/direction_comparison.json \
    --layers_of_interest 12,17,31 --n_boot 1000 --seed 42

echo "Done at $(date)"
