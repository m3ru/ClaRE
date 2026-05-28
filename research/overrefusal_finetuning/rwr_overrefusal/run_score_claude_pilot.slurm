#!/bin/bash
#SBATCH --job-name=claude-pilot-score
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/claude_pilot_score_%j.out
#SBATCH --error=logs/claude_pilot_score_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Score a Claude-generated dataset JSONL using the existing score_candidates.py
# (which lives at the same level as this slurm script). Output is shard JSONs
# in <output_dir>, in the same format consumed by rwr_data.load_shards.
#
# Submit:
#   sbatch --export=ALL,INPUT_JSONL=<path>,OUTPUT_DIR=<path> run_score_claude_pilot.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

INPUT_JSONL="${INPUT_JSONL:-prompt_iteration_results/dataset_pilot.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-prompt_iteration_results/dataset_pilot_shards}"

echo "Starting pilot score at $(date)"
echo "Input:  ${INPUT_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u score_candidates.py \
    --input_jsonl "${INPUT_JSONL}" \
    --output_dir  "${OUTPUT_DIR}" \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
    --target_model meta-llama/Meta-Llama-3-8B-Instruct \
    --activation_layer 32 \
    --batch_size 8 \
    --shard_size 50 \
    --similarity_exponent 5.0 \
    --similarity_center 0.75 \
    --refusal_divisor 100.0

echo "Pilot score completed at $(date)"
ls -la "${OUTPUT_DIR}" || true
