#!/bin/bash
#SBATCH --job-name=probe-pair-extract
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/probe_pair_extract_%j.out
#SBATCH --error=logs/probe_pair_extract_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs probe_or/pair_acts

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Paired activation extraction at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u probe_or/extract_pair_acts.py \
    --pairs_csv probe_or/pairs.csv \
    --out_dir probe_or/pair_acts

echo "Done at $(date)"
echo "Next (login): python probe_or/probe_delta.py --acts_dir probe_or/activations --pair_dir probe_or/pair_acts --direction lda"
