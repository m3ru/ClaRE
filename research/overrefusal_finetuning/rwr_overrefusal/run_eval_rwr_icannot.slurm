#!/bin/bash
#SBATCH --job-name=eval-rwr-icannot
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/eval_rwr_icannot_%j.out
#SBATCH --error=logs/eval_rwr_icannot_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs prompt_iteration_results/eval_icannot

module load gcc arrow/23.0.1 cuda scipy-stack/2024b   # arrow: sentence_transformers -> datasets -> pyarrow
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Evaluating RWR (icannot-OR) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u eval_rwr_icannot.py \
    --adapter_dir rwr_icannot_checkpoints/final \
    --shard_dir icannot_train_shards \
    --output prompt_iteration_results/eval_icannot/eval_icannot_results.json \
    --n_per_prompt 5 --temperature 0.7 --max_eval_prompts 200 \
    --eval_base_model

echo "Eval complete at $(date)"
