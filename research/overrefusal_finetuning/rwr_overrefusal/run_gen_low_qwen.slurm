#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --job-name=gen-low-qwen
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/gen_b2_qwen_%A_%a.out
#SBATCH --error=logs/gen_b2_qwen_%A_%a.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "gen batch2 qwen shard ${SLURM_ARRAY_TASK_ID} at $(date)"
python -u generate_or_alpaca.py \
    --adapter_dir probe_or/results/rwr_qwen_ckpt_probe/final \
    --base_model Qwen/Qwen3-32B \
    --output_dir probe_or/results/gen_low_qwen_probe \
    --shuffle_seed 43 --skip_first 14000 --num_prompts 32000 \
    --num_shards 16 --shard_idx ${SLURM_ARRAY_TASK_ID} \
    --n_per_prompt 4 --temperature 0.7 --top_p 0.9 --max_new_tokens 256 --batch_size 8
echo "done shard ${SLURM_ARRAY_TASK_ID} at $(date)"
