#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=sim-scaleup-llama
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=logs/sim_scaleup_%j.out
#SBATCH --error=logs/sim_scaleup_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u build_scaleup_corpus.py \
  --gen_dir probe_or/results/gen_scaleup_llama_logit \
  --glob "gen_alpaca_or_shard*.json" \
  --out_csv probe_or/results/scaleup_atlas_llama/corpus_scaleup_llama.csv \
  --exclude_json $HOME/.claude/jobs/2dc7b27c/tmp/scaleup_exclude.json
echo "[sim] done at $(date)"
