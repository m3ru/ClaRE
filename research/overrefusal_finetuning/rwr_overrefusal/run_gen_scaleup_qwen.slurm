#!/bin/bash
#SBATCH --job-name=gen-scaleup-qwen
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/gen_scaleup_qwen_%A_%a.out
#SBATCH --error=logs/gen_scaleup_qwen_%A_%a.err
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Scale-up rewrite generation with the chosen Qwen attacker (rwr_qwen_ckpt_logit/final).
# SAME disjoint prompt set as the Llama scale-up: seed-43 shuffle, skip_first 6000, 2000
# prompts -> VERIFIED disjoint from the Qwen pool (5,998 originals == the Llama pool) and
# the 599 Qwen held-out (0 overlap with the parent's actual 2,000 scale-up prompts). So the
# Llama and Qwen corpora share originals and are directly comparable.
# Full 80GB H100 (bf16 32B ~64GB; MIG 40GB slice would OOM). enable_thinking=False fix in
# generate_or_alpaca.py is required for Qwen3 (else <think> blocks corrupt the rewrites).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/gen_scaleup_qwen_logit

module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "gen scale-up qwen shard ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u generate_or_alpaca.py \
    --adapter_dir probe_or/results/rwr_qwen_ckpt_logit/final \
    --base_model Qwen/Qwen3-32B \
    --output_dir probe_or/results/gen_scaleup_qwen_logit \
    --shuffle_seed 43 --skip_first 6000 --num_prompts 2000 \
    --num_shards 4 --shard_idx ${SLURM_ARRAY_TASK_ID} \
    --n_per_prompt 4 --temperature 0.7 --top_p 0.9 --max_new_tokens 256 --batch_size 8

echo "done shard ${SLURM_ARRAY_TASK_ID} at $(date)"
