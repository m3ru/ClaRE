#!/bin/bash
#SBATCH --job-name=eval-rwr-qwen
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/eval_rwr_qwen_%j.out
#SBATCH --error=logs/eval_rwr_qwen_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/eval_qwen

# arrow: eval_rwr_qwen -> sentence_transformers -> datasets -> pyarrow
module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Behavioral eval of RWR-Qwen at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

CKPT=probe_or/results/rwr_qwen_ckpt
# final vs base-Qwen-as-attacker (same held-out split as training)
python -u eval_rwr_qwen.py \
    --adapter_dir "$CKPT/final" --shard_dir probe_or/results/qwen_shards \
    --output probe_or/results/eval_qwen/eval_final.json \
    --base_model Qwen/Qwen3-32B \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200 --eval_base_model

# epoch 2 (memorization check; base omitted -- reuse from final eval)
python -u eval_rwr_qwen.py \
    --adapter_dir "$CKPT/epoch_2" --shard_dir probe_or/results/qwen_shards \
    --output probe_or/results/eval_qwen/eval_epoch2.json \
    --base_model Qwen/Qwen3-32B \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200

echo "Eval complete at $(date)"
