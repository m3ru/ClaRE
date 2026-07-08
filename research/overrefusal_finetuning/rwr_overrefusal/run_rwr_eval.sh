#!/bin/bash
#SBATCH --job-name=rwr-eval
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/rwr_eval_%j.out
#SBATCH --error=logs/rwr_eval_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting RWR eval at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -u eval_rwr.py \
    --adapter_dir ./rwr_checkpoints/final \
    --shard_dir ../or_paraphrase_3k,./scored_taskaware \
    --refusal_vector_path ../../refusal_vector/3_Vector_Extraction/refusal_vector.layer032.npz \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --eval_base_model \
    --n_per_prompt 3 \
    --max_eval_prompts 200 \
    --temperature 0.7 \
    --output eval_results.json

echo "RWR eval completed at $(date)"
