#!/bin/bash
#SBATCH --job-name=gen-scaleup-llama
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/gen_scaleup_llama_%A_%a.out
#SBATCH --error=logs/gen_scaleup_llama_%A_%a.err
#SBATCH --export=ALL

# Scale-up rewrite generation with the WINNING logit attacker
# (probe_or/results/rwr_llama_ckpt_logit/final). Pool = the seed-43 shuffle of
# remaining_prompts()[6000:] -> provably disjoint from the Sonnet training pool
# (first 6000), the 599 held-out originals, and all train/iter/eval seeds.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/gen_scaleup_llama_logit

module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "gen scale-up shard ${SLURM_ARRAY_TASK_ID} at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u generate_or_alpaca.py \
    --adapter_dir probe_or/results/rwr_llama_ckpt_logit/final \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir probe_or/results/gen_scaleup_llama_logit \
    --shuffle_seed 43 --skip_first 6000 --num_prompts 2000 \
    --num_shards 4 --shard_idx ${SLURM_ARRAY_TASK_ID} \
    --n_per_prompt 4 --temperature 0.7 --top_p 0.9 --max_new_tokens 256 --batch_size 16

echo "done shard ${SLURM_ARRAY_TASK_ID} at $(date)"
