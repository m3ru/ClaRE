#!/bin/bash
#SBATCH --job-name=rwr-eval
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/rwr_eval_%j.out
#SBATCH --error=logs/rwr_eval_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Build a clean venv on the node's local disk (fast I/O, avoids parallel FS issues)
module load cuda scipy-stack/2024b
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

export HF_HOME=~/.cache/huggingface

echo "Starting RWR eval at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -u eval_rwr.py \
    --adapter_dir ./rwr_checkpoints/final \
    --shard_dir ../or_paraphrase_3k \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --eval_base_model \
    --n_per_prompt 5 \
    --temperature 0.7 \
    --output eval_results.json

echo "RWR eval completed at $(date)"
