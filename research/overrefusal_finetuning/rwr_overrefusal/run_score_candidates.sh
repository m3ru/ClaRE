#!/bin/bash
#SBATCH --job-name=rwr-score
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/rwr_score_%j.out
#SBATCH --error=logs/rwr_score_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting candidate scoring at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -u score_candidates.py \
    --input_jsonl ../../../overrefusal_sampling/outputs/taskaware_5k.jsonl \
    --output_dir ./scored_taskaware \
    --refusal_vector_path ../../refusal_vector/3_Vector_Extraction/refusal_vector.layer032.npz \
    --target_model meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 8 \
    --similarity_exponent 5.0 \
    --similarity_center 0.75 \
    --refusal_divisor 100.0

echo "Scoring completed at $(date)"
