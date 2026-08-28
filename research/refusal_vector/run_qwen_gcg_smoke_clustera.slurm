#!/bin/bash
#SBATCH --job-name=qwen-gcg-smoke
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/qwen_gcg_smoke_%A_%a.out
#SBATCH --error=logs/qwen_gcg_smoke_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=$REPO
PAIR_ROOT="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/probe_or"
OPENER_JSON="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/refusal_atlas/opener_sets.json"
OUT=$PROJECT/qwen32b_gcg_v1/smoke
mkdir -p "${OUT}"

COMMON=(
  --pairs_csv "${PAIR_ROOT}/pairs.csv" "${PAIR_ROOT}/pairs_sonnet.csv"
  --base_model Qwen/Qwen3-32B --disable_thinking --attn_implementation eager
  --opener_json "${OPENER_JSON}" --model_key qwen --objective openers
  --n_prompts 1 --steps 3 --n_cand 16 --candidate_batch_size 16
  --sim_floor 0.85 --lam 20.0 --block_profanity
  --fluency_weight 1.0 --nll_max_weight 0.5 --max_new_tokens 32
  --generation_batch_size 1
)

if [[ "${SLURM_ARRAY_TASK_ID}" == 0 ]]; then
  python -u "${ROOT}/research/refusal_vector/prompt_rewrite_opt.py" \
    "${COMMON[@]}" --out "${OUT}/loose.json" \
    --topk 256 --lm_topk_filter 512 --extra_tokens 12 --whole_word_only
else
  python -u "${ROOT}/research/refusal_vector/prompt_rewrite_opt.py" \
    "${COMMON[@]}" --out "${OUT}/strict.json" \
    --topk 64 --lm_topk_filter 64 --editable_single_token_words_only --max_edits 4 \
    --max_nll_increase 0.15 --max_nll_max_increase 1.0
fi

echo "[done] Qwen smoke arm ${SLURM_ARRAY_TASK_ID} $(date)"
