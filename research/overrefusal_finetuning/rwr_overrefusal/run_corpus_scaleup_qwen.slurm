#!/bin/bash
#SBATCH --job-name=corpus-scaleup-qwen
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-00:40:00
#SBATCH --output=logs/corpus_scaleup_qwen_%j.out
#SBATCH --error=logs/corpus_scaleup_qwen_%j.err
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Consolidate the Qwen attacker rewrite shards -> flat corpus CSV with MiniLM similarity
# (pair_id, orig_idx, original, rewrite, similarity). Model-agnostic build_scaleup_corpus.py,
# same MiniLM metric the RWR reward/eval used. Disjointness guard excludes the Qwen held-out
# (already 0 overlap by construction; belt-and-suspenders).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

python -u build_scaleup_corpus.py \
    --gen_dir probe_or/results/gen_scaleup_qwen_logit \
    --glob 'gen_*_shard*.json' \
    --exclude_json probe_or/results/qwen_scored/heldout_originals.json \
    --out_csv probe_or/results/scaleup_corpus_qwen_logit.csv

echo "corpus built at $(date)"
