#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=gen-base-llama
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/gen_base_llama_%A_%a.out
#SBATCH --error=logs/gen_base_llama_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u generate_or_alpaca.py \
    --adapter_dir none \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir probe_or/results/gen_base_llama \
    --shuffle_seed 43 --skip_first 8000 --num_prompts 6000 \
    --num_shards 8 --shard_idx ${SLURM_ARRAY_TASK_ID} \
    --n_per_prompt 4 --temperature 0.7 --top_p 0.9 --max_new_tokens 256 --batch_size 16
