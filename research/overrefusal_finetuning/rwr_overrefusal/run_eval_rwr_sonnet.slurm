#!/bin/bash
#SBATCH --job-name=eval-rwr-sonnet
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/eval_rwr_sonnet_%j.out
#SBATCH --error=logs/eval_rwr_sonnet_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs logprob_or/results/eval_sonnet

# arrow: eval_rwr_icannot -> sentence_transformers -> datasets -> pyarrow
module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Evaluating RWR-Sonnet (icannot-OR) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

CKPT=logprob_or/results/rwr_sonnet_ckpt
# final vs base (same held-out split as training: shard_dir + seed + val_fraction)
python -u eval_rwr_icannot.py \
    --adapter_dir "$CKPT/final" --shard_dir logprob_or/results/shards_sonnet \
    --output logprob_or/results/eval_sonnet/eval_final.json \
    --n_per_prompt 5 --temperature 0.7 --max_eval_prompts 200 --eval_base_model

# epoch 2 (memorization check; base omitted -- reuse base numbers from the final eval)
python -u eval_rwr_icannot.py \
    --adapter_dir "$CKPT/epoch_2" --shard_dir logprob_or/results/shards_sonnet \
    --output logprob_or/results/eval_sonnet/eval_epoch2.json \
    --n_per_prompt 5 --temperature 0.7 --max_eval_prompts 200

echo "Eval complete at $(date)"
