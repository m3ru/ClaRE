#!/bin/bash
# <cluster> fallback when the persistent 300-GB scratch quota cannot hold Qwen3-32B.
# Downloads the model once to node-local /tmp, then runs loose and strict smoke
# arms concurrently on two GPUs. The cache is removed automatically at job exit.
#SBATCH --job-name=qwen-gcg-smoke
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu-h200|gpu-h100"
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=qwen32b_gcg_v1/logs/smoke_%j.out
#SBATCH --error=qwen32b_gcg_v1/logs/smoke_%j.err

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOME}/scratch/conda_envs/ppo-or-env"

ROOT="${HOME}/ClaRE"
PAIR_ROOT="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/probe_or"
OPENER_JSON="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/refusal_atlas/opener_sets.json"
OUT="${HOME}/qwen32b_gcg_v1/smoke"
NODE_CACHE="/tmp/qwen32b_gcg_${SLURM_JOB_ID}"
mkdir -p "${OUT}" "${NODE_CACHE}"
trap 'rm -rf "/tmp/qwen32b_gcg_${SLURM_JOB_ID}"' EXIT

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_HOME="${NODE_CACHE}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
unset HF_HUB_DISABLE_XET HF_XET_DISABLE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[stage] Qwen3-32B and MiniLM to ${NODE_CACHE} $(date)"
hf download Qwen/Qwen3-32B
hf download sentence-transformers/all-MiniLM-L6-v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

COMMON=(
  --pairs_csv "${PAIR_ROOT}/pairs.csv" "${PAIR_ROOT}/pairs_sonnet.csv"
  --base_model Qwen/Qwen3-32B --disable_thinking --attn_implementation eager
  --opener_json "${OPENER_JSON}" --model_key qwen --objective openers
  --n_prompts 1 --steps 3 --n_cand 16 --candidate_batch_size 16
  --sim_floor 0.85 --lam 20.0 --block_profanity
  --fluency_weight 1.0 --nll_max_weight 0.5 --max_new_tokens 32
  --generation_batch_size 1
)

echo "[run] parallel loose+strict smoke $(date)"
CUDA_VISIBLE_DEVICES=0 python -u "${ROOT}/research/refusal_vector/prompt_rewrite_opt.py" \
  "${COMMON[@]}" --out "${OUT}/loose.json" \
  --topk 256 --lm_topk_filter 512 --extra_tokens 12 --whole_word_only \
  >"${OUT}/loose.out" 2>"${OUT}/loose.err" &
PID_LOOSE=$!

CUDA_VISIBLE_DEVICES=1 python -u "${ROOT}/research/refusal_vector/prompt_rewrite_opt.py" \
  "${COMMON[@]}" --out "${OUT}/strict.json" \
  --topk 64 --lm_topk_filter 64 --editable_single_token_words_only --max_edits 4 \
  --max_nll_increase 0.15 --max_nll_max_increase 1.0 \
  >"${OUT}/strict.out" 2>"${OUT}/strict.err" &
PID_STRICT=$!

STATUS=0
wait "${PID_LOOSE}" || STATUS=$?
wait "${PID_STRICT}" || STATUS=$?
[[ "${STATUS}" == 0 ]] || { echo "smoke failed; inspect ${OUT}/*.err" >&2; exit "${STATUS}"; }

echo "[done] Qwen BF16 smoke $(date)"
