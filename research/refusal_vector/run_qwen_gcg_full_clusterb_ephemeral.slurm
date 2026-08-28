#!/bin/bash
# One Qwen3-32B GCG shard per GPU. <cluster>'s <QOS> QOS limits each job to
# 960 GPU-minutes, so the 3,200-prompt OR arms use 40 x 80-prompt shards.
# Jobs on the same physical node share an immutable /tmp Hugging Face cache.
# Submit OR arms with the default array 0-39; submit jailbreak arms with
# --array=0-7 (eight shards x up to 28 prompts).
#SBATCH --job-name=qwen-gcg-full
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100"
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-39
#SBATCH --output=$HOME/qwen32b_gcg_v1/logs/full_%A_%a.out
#SBATCH --error=$HOME/qwen32b_gcg_v1/logs/full_%A_%a.err

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
NODE_CACHE="/tmp/qwen32b_gcg_shared_${USER}"
S="${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUT}" "${NODE_CACHE}" "${RUN_ROOT}/logs"

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_HOME="${NODE_CACHE}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
unset HF_HUB_DISABLE_XET HF_XET_DISABLE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[stage] ${ARM} shard ${S}: Qwen3-32B + MiniLM on $(hostname) $(date)"
# hf_hub's lock files make simultaneous first-use downloads to a shared node
# cache safe; subsequent shards on the node resolve this immediately.
hf download Qwen/Qwen3-32B
hf download sentence-transformers/all-MiniLM-L6-v2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

COMMON=(
  --base_model Qwen/Qwen3-32B --disable_thinking --attn_implementation eager
  --opener_json "${OPENER_JSON}" --model_key qwen
  --steps 150 --n_cand 96 --candidate_batch_size 16
  --sim_floor 0.85 --lam 20.0 --block_profanity
  --fluency_weight 1.0 --nll_max_weight 0.5
  --max_new_tokens 128 --generation_batch_size 8
)

if [[ "${ARM}" == or_* ]]; then
  COMMON+=(
    --pairs_csv "${PAIR_ROOT}/pairs.csv" "${PAIR_ROOT}/pairs_sonnet.csv"
    --objective openers --shard_id "${S}" --n_shards 40 --n_prompts 80
    --holdout_frac 0.30 --max_prompt_tok 48
    --dump_holdout "${OUT}/holdout_s${S}.json"
  )
  STEM=qwen_or
else
  [[ -s "${HARMFUL_CSV}" ]] || { echo "missing ${HARMFUL_CSV}" >&2; exit 2; }
  COMMON+=(
    --prompts_csv "${HARMFUL_CSV}" --prompt_column prompt
    --objective anti_openers --shard_id "${S}" --n_shards 8 --n_prompts 28
    --max_prompt_tok 96
  )
  STEM=qwen_jb
fi

if [[ "${ARM}" == *_loose ]]; then
  SEARCH=(--topk 256 --lm_topk_filter 512 --extra_tokens 12 --whole_word_only)
else
  SEARCH=(
    --topk 64 --lm_topk_filter 64 --editable_single_token_words_only --max_edits 4
    --max_nll_increase 0.15 --max_nll_max_increase 1.0
  )
fi

echo "[run] ${ARM} shard ${S} $(date)"
python -u "${ROOT}/research/refusal_vector/prompt_rewrite_opt.py" \
  "${COMMON[@]}" "${SEARCH[@]}" --out "${OUT}/${STEM}_s${S}.json" \
  >"${OUT}/${STEM}_s${S}.out" 2>"${OUT}/${STEM}_s${S}.err"

echo "[done] ${ARM} shard ${S} $(date)"
