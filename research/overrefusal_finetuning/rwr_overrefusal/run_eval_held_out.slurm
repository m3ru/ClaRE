#!/bin/bash
#SBATCH --job-name=rwr-eval-held-out
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/rwr_eval_%j.out
#SBATCH --error=logs/rwr_eval_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Plan B held-out eval: baseline Llama vs Claude-RWR vs RWR v3 on
# held-out alpaca + held-out dolly. Submit with:
#
#   sbatch --export=ALL,CLAUDE_RWR=./rwr_claude_checkpoints/final,RWR_V3=<path> run_eval_held_out.slurm
#
# Pass an empty RWR_V3 to skip the v3 baseline; pass empty CLAUDE_RWR to skip the
# Claude-trained model. Baseline is always included.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CLAUDE_RWR="${CLAUDE_RWR:-./rwr_claude_checkpoints/final}"
RWR_V3="${RWR_V3:-}"

# Build the --models arg list incrementally so we can skip any model with an empty path.
MODELS=("baseline=baseline")
[ -n "${CLAUDE_RWR}" ] && [ -d "${CLAUDE_RWR}" ] && MODELS+=("claude_rwr=${CLAUDE_RWR}")
[ -n "${RWR_V3}"     ] && [ -d "${RWR_V3}"     ] && MODELS+=("rwr_v3=${RWR_V3}")

echo "Starting held-out eval at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Models: ${MODELS[@]}"

python -u eval_held_out.py \
    --models "${MODELS[@]}" \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --dolly_csv ../../overrefusal_sampling/data/benign_dolly.csv \
    --output_dir prompt_iteration_results/held_out_eval \
    --n_alpaca 200 \
    --n_dolly 200 \
    --n_per_prompt 3 \
    --temperature 0.7 \
    --max_new_tokens 64 \
    --gen_batch_size 4 \
    --score_batch_size 8 \
    --eval_seed 99 \
    --training_seed 42 \
    --training_n_alpaca 2500 \
    --training_n_dolly 2500 \
    --iter_seed 42 \
    --iter_n 20

echo "Held-out eval completed at $(date)"
