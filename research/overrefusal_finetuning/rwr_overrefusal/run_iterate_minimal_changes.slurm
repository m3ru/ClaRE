#!/bin/bash
#SBATCH --job-name=prompt-iter-min
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/prompt_iter_min_%j.out
#SBATCH --error=logs/prompt_iter_min_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting minimal_changes iteration at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u iterate_prompts_alpaca.py \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir ./prompt_iteration_results \
    --output_name minimal_changes_v2_results.json \
    --variant_names minimal_changes \
    --no_baseline \
    --n_prompts 20 \
    --n_per_prompt 3 \
    --temperature 0.7 \
    --max_new_tokens 64 \
    --score_batch_size 8 \
    --seed 42

echo "minimal_changes iteration completed at $(date)"
