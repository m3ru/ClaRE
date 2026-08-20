#!/bin/bash
# Eight-GPU packed Qwen3-32B GCG run. Each node downloads the model once to
# local /tmp and runs one independent shard per GPU. Submit OR arms with the
# default array 0-4 (40 shards); submit jailbreak arms with --array=0 and
# N_SHARDS=8.
#SBATCH --job-name=qwen-gcg-full
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:8
#SBATCH --constraint="gpu-h200|gpu-h100"
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/hice1/aharris345/qwen32b_gcg_v1/logs/full_%A_%a.out
#SBATCH --error=/home/hice1/aharris345/qwen32b_gcg_v1/logs/full_%A_%a.err

set -euo pipefail
: "${ARM:?submit with ARM=or_loose|or_strict|jb_loose|jb_strict}"
case "${ARM}" in
  or_loose|or_strict|jb_loose|jb_strict) ;;
  *) echo "invalid ARM=${ARM}" >&2; exit 2 ;;
esac

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOME}/scratch/conda_envs/ppo-or-env"

ROOT="${HOME}/ClaRE"
PAIR_ROOT="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/probe_or"
OPENER_JSON="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/refusal_atlas/opener_sets.json"
RUN_ROOT="${HOME}/qwen32b_gcg_v1"
HARMFUL_CSV="${RUN_ROOT}/data/strongreject_custom_direct.csv"
OUT="${RUN_ROOT}/${ARM}"
NODE_CACHE="/tmp/qwen32b_gcg_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUT}" "${NODE_CACHE}" "${RUN_ROOT}/logs"
trap 'rm -rf "/tmp/qwen32b_gcg_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' EXIT

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_HOME="${NODE_CACHE}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
unset HF_HUB_DISABLE_XET HF_XET_DISABLE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[stage] ${ARM} task ${SLURM_ARRAY_TASK_ID}: Qwen3-32B + MiniLM $(date)"
hf download Qwen/Qwen3-32B
hf download sentence-transformers/all-MiniLM-L6-v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

if [[ "${ARM}" == or_* ]]; then
  N_SHARDS="${N_SHARDS:-40}"
  N_PROMPTS="${N_PROMPTS:-80}"
  STEM=qwen_or
else
  N_SHARDS="${N_SHARDS:-8}"
  N_PROMPTS="${N_PROMPTS:-28}"
  STEM=qwen_jb
  [[ -s "${HARMFUL_CSV}" ]] || { echo "missing ${HARMFUL_CSV}" >&2; exit 2; }
fi

BASE_SHARD=$((SLURM_ARRAY_TASK_ID * 8))
if (( BASE_SHARD + 7 >= N_SHARDS )); then
  echo "task ${SLURM_ARRAY_TASK_ID} maps past N_SHARDS=${N_SHARDS}" >&2
  exit 2
fi

PIDS=()
for LOCAL_RANK in $(seq 0 7); do
  S=$((BASE_SHARD + LOCAL_RANK))
  COMMON=(
    --base_model Qwen/Qwen3-32B --disable_thinking --attn_implementation eager
    --opener_json "${OPENER_JSON}" --model_key qwen
    --shard_id "${S}" --n_shards "${N_SHARDS}" --steps 150
    --n_cand 96 --candidate_batch_size 16
    --sim_floor 0.85 --lam 20.0 --block_profanity
    --fluency_weight 1.0 --nll_max_weight 0.5
    --max_new_tokens 128 --generation_batch_size 8 --n_prompts "${N_PROMPTS}"
  )

  if [[ "${ARM}" == or_* ]]; then
    COMMON+=(
      --pairs_csv "${PAIR_ROOT}/pairs.csv" "${PAIR_ROOT}/pairs_sonnet.csv"
      --objective openers --holdout_frac 0.30 --max_prompt_tok 48
      --dump_holdout "${OUT}/holdout_s${S}.json"
    )
  else
    COMMON+=(
      --prompts_csv "${HARMFUL_CSV}" --prompt_column prompt
      --objective anti_openers --max_prompt_tok 96
    )
  fi

  if [[ "${ARM}" == *_loose ]]; then
    SEARCH=(--topk 256 --lm_topk_filter 512 --extra_tokens 12 --whole_word_only)
  else
    SEARCH=(
      --topk 64 --lm_topk_filter 64 --editable_single_token_words_only --max_edits 4
      --max_nll_increase 0.15 --max_nll_max_increase 1.0
    )
  fi

  echo "[launch] ${ARM} shard ${S} on local GPU ${LOCAL_RANK} $(date)"
  CUDA_VISIBLE_DEVICES="${LOCAL_RANK}" python -u \
    "${ROOT}/research/refusal_vector/prompt_rewrite_opt.py" \
    "${COMMON[@]}" "${SEARCH[@]}" --out "${OUT}/${STEM}_s${S}.json" \
    >"${OUT}/${STEM}_s${S}.out" 2>"${OUT}/${STEM}_s${S}.err" &
  PIDS+=("$!")
  sleep 2
done

STATUS=0
for PID in "${PIDS[@]}"; do
  wait "${PID}" || STATUS=$?
done
[[ "${STATUS}" == 0 ]] || { echo "one or more shards failed: status=${STATUS}" >&2; exit "${STATUS}"; }

echo "[done] ${ARM} task ${SLURM_ARRAY_TASK_ID} $(date)"
