#!/bin/bash
#SBATCH --job-name=rwr-eval-bucketing
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/rwr_eval_bucketing_%j.out
#SBATCH --error=logs/rwr_eval_bucketing_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

# Head-to-head: baseline + claude_rwr (original) + claude_rwr_v1 (finer top bins)
# on the same 200 held-out alpaca + 200 held-out dolly prompts.
#
# Override paths via env vars if checkpoint dirs differ:
#   sbatch --export=ALL,CLAUDE_RWR_V1=./rwr_claude_v1_checkpoints/final \
#          run_eval_bucketing_compare.slurm
#
# After this finishes, re-score at k=5.0 to be directly comparable to published
# claude_rwr numbers (claude_training_results.md):
#   python -u rescore_held_out.py \
#       --input  prompt_iteration_results/held_out_eval_bucketing/held_out_eval_results.json \
#       --output prompt_iteration_results/held_out_eval_bucketing/held_out_eval_results_k5.json

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
CLAUDE_RWR_V1="${CLAUDE_RWR_V1:-./rwr_claude_v1_checkpoints/final}"

MODELS=("baseline=baseline")
[ -n "${CLAUDE_RWR}"    ] && [ -d "${CLAUDE_RWR}"    ] && MODELS+=("claude_rwr=${CLAUDE_RWR}")
[ -n "${CLAUDE_RWR_V1}" ] && [ -d "${CLAUDE_RWR_V1}" ] && MODELS+=("claude_rwr_v1=${CLAUDE_RWR_V1}")

echo "Starting bucketing head-to-head eval at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Models: ${MODELS[@]}"

python -u eval_held_out.py \
    --models "${MODELS[@]}" \
    --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --dolly_csv ../../overrefusal_sampling/data/benign_dolly.csv \
    --output_dir prompt_iteration_results/held_out_eval_bucketing \
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

echo "Bucketing eval completed at $(date)"
