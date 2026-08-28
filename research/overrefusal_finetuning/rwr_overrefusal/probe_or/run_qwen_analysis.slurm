#!/bin/bash
#SBATCH --job-name=qwen-signals
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/qwen_signals_%j.out
#SBATCH --error=logs/qwen_signals_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/qwen_signals

module load scipy-stack/2024b          # numpy/scipy; sklearn from ~general
source ~/general/bin/activate

echo "Qwen signal build (vector + probe ensemble) at $(date)"

for DIR in raw lda; do
  echo "===== direction=$DIR ====="
  python -u probe_or/probe_qwen_signals.py \
      --acts_dir probe_or/qwen_activations \
      --pair_dir probe_or/qwen_pair_acts \
      --behav_csv probe_or/results/qwen_behav/behav.csv \
      --out_dir probe_or/results/qwen_signals \
      --direction "$DIR" --dP_thresh 0.01
done

echo "Done at $(date)"
