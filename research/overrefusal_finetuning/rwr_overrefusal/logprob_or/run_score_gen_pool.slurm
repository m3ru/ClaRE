#!/bin/bash
#SBATCH --job-name=score-gen-sonnet
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/score_gen_pool_%j.out
#SBATCH --error=logs/score_gen_pool_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs logprob_or/results/icannot_or_sonnet

# arrow: sentence_transformers -> datasets -> pyarrow (learned from the eval crash)
module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Scoring Sonnet-5 gen pool with icannot-OR at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u logprob_or/score_gen_pool.py \
    --gen_dir gen_or_sonnet \
    --out_dir logprob_or/results/icannot_or_sonnet

echo "Done at $(date)"
