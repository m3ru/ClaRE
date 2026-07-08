#!/bin/bash
#SBATCH --job-name=rwr-rescore
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/rwr_rescore_%j.out
#SBATCH --error=logs/rwr_rescore_%j.err
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

echo "Starting held-out re-score at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u rescore_held_out.py \
    --input prompt_iteration_results/held_out_eval/held_out_eval_results.json \
    --output prompt_iteration_results/held_out_eval/held_out_eval_results_k5.json \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz

echo "Re-score completed at $(date)"
