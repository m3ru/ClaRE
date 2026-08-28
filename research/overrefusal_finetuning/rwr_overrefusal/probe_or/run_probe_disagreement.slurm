#!/bin/bash
#SBATCH --job-name=probe-disagree
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/probe_disagree_%j.out
#SBATCH --error=logs/probe_disagree_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs probe_or/results

module load cuda scipy-stack/2024b          # transformers + numpy/scipy; no arrow needed
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "(B) probe-vs-logprob disagreement at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 200 KEY + 150 CTRL + 50 POS = 400 rewrites x 5 samples = 2000 generations x 64 tok.
python -u probe_or/probe_disagreement.py \
    --pairs_csv probe_or/pairs_sonnet.csv \
    --pair_dir probe_or/pair_acts_sonnet \
    --acts_dir probe_or/activations \
    --out probe_or/results/disagreement.json \
    --n_key 200 --n_ctrl 150 --n_pos 50 \
    --n_samples 5 --temperature 0.7 --max_new_tokens 64

echo "Done at $(date)"
