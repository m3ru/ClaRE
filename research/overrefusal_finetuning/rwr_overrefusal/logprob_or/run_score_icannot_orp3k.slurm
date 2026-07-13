#!/bin/bash
#SBATCH --job-name=icannot-orp3k
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/icannot_orp3k_%j.out
#SBATCH --error=logs/icannot_orp3k_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir (parent of logprob_or/)
mkdir -p logs

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting icannot-OR scoring of orp3k at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u logprob_or/score_icannot_or.py \
    --shard_dir ../or_paraphrase_3k_deduped \
    --out_dir logprob_or/results/icannot_or_orp3k

echo "Completed at $(date)"
