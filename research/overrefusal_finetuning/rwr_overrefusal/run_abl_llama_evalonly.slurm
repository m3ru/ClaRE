#!/bin/bash
#SBATCH --job-name=abl-llama-eval
#SBATCH --account=def-vganesh
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/abl_llama_eval_%j.out
#SBATCH --error=logs/abl_llama_eval_%j.err
#SBATCH --export=ALL
# Stage 4 only. Job 19410996 scored, built shards and TRAINED successfully, then died
# importing sentence_transformers because pyarrow needs `gcc arrow/23.0.1` (CLAUDE.md).
# The adapter is on disk, so only the eval needs re-running.
set -euo pipefail
mkdir -p probe_or/results/eval_llama_vector_abl_L12
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import pyarrow, sentence_transformers; print('[pre-flight] pyarrow + sentence_transformers import OK')"
python -u eval_rwr_llama.py \
    --adapter_dir probe_or/results/rwr_llama_ckpt_vector_abl_L12/final \
    --heldout_json probe_or/results/llama_scored/heldout_originals.json \
    --output probe_or/results/eval_llama_vector_abl_L12/eval_final.json \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200 --eval_base_model
echo "### done $(date)"
