#!/bin/bash
#SBATCH --job-name=claude-score
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/claude_score_%j.out
#SBATCH --error=logs/claude_score_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Score phase only. Generation must already have run on a login node:
#   ANTHROPIC_API_KEY=sk-... python iterate_prompts_claude.py --skip_scoring --output_name <X>
# Then:
#   sbatch --export=ALL,OUTPUT_NAME=<X> run_score_claude.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUTPUT_NAME="${OUTPUT_NAME:-claude_results.json}"

echo "Starting claude score phase at $(date) (output_name=${OUTPUT_NAME})"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u iterate_prompts_claude.py \
    --skip_generation \
    --output_name "${OUTPUT_NAME}" \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
    --scoring_model meta-llama/Meta-Llama-3-8B-Instruct \
    --score_batch_size 8

echo "claude score phase completed at $(date)"
