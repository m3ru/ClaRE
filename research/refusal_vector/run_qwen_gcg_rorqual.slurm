#!/bin/bash
# Submit one array per arm, for example:
#   sbatch --export=ALL,ARM=or_loose run_qwen_gcg_rorqual.slurm
#   sbatch --export=ALL,ARM=jb_loose --array=0-7 run_qwen_gcg_rorqual.slurm
#
# OR_NPROMPTS sets the prompts per OR shard (default 80 => 3,200 total). Lower it
# to trade statistical power for wall clock when the cluster is contended, e.g.
# OR_NPROMPTS=24 gives 960 prompts per arm (+-3.2% vs +-1.7% at 95%) in ~2.7 h
# per shard instead of ~9.1 h:
#   sbatch --export=ALL,ARM=or_loose,OR_NPROMPTS=24 run_qwen_gcg_rorqual.slurm
# Valid ARM values: or_loose, or_strict, jb_loose, jb_strict.
#
# Measured on a rorqual H100: ~386 s/prompt loose, ~334 s/prompt strict at
# these search settings, and ~65 GB of the 80 GB card. prompt_rewrite_opt.py
# writes its JSON only after the last prompt, so a shard that overruns its wall
# clock loses the whole shard -- hence 80 prompts (~8.6 h) against a 12 h limit.
# The OR arms therefore use the default 40 x 80 array; submit the jailbreak arms
# with --array=0-7 (eight shards covering all 221 StrongREJECT prompts).
#SBATCH --job-name=qwen-gcg
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-12:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/qwen_gcg_%A_%a.out
#SBATCH --error=logs/qwen_gcg_%A_%a.err

set -euo pipefail
: "${ARM:?submit with --export=ALL,ARM=or_loose|or_strict|jb_loose|jb_strict}"
case "${ARM}" in
  or_loose|or_strict|jb_loose|jb_strict) ;;
  *) echo "invalid ARM=${ARM}" >&2; exit 2 ;;
esac

echo "[cfg] ARM=${ARM} OR_NPROMPTS=${OR_NPROMPTS:-80} shard=${SLURM_ARRAY_TASK_ID}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/home/meru/links/scratch/ClaRE
RUN_ROOT=/home/meru/links/projects/def-vganesh/meru/qwen32b_gcg_v1
PAIR_ROOT="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/probe_or"
OPENER_JSON="${ROOT}/research/overrefusal_finetuning/rwr_overrefusal/refusal_atlas/opener_sets.json"
HARMFUL_CSV="${RUN_ROOT}/data/strongreject_custom_direct.csv"
OUT="${RUN_ROOT}/${ARM}"
S="${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUT}"

COMMON=(
  --base_model Qwen/Qwen3-32B --disable_thinking --attn_implementation eager
  --opener_json "${OPENER_JSON}" --model_key qwen
  --shard_id "${S}" --steps 150 --n_cand 96 --candidate_batch_size 16
  --sim_floor 0.85 --lam 20.0 --block_profanity
  --fluency_weight 1.0 --nll_max_weight 0.5
  --max_new_tokens 128 --generation_batch_size 8
)

if [[ "${ARM}" == or_* ]]; then
  COMMON+=(
    --pairs_csv "${PAIR_ROOT}/pairs.csv" "${PAIR_ROOT}/pairs_sonnet.csv"
    --objective openers --n_shards 40 --n_prompts "${OR_NPROMPTS:-80}"
    --holdout_frac 0.30 --max_prompt_tok 48
    --dump_holdout "${OUT}/holdout_s${S}.json"
  )
  STEM=qwen_or
else
  [[ -s "${HARMFUL_CSV}" ]] || { echo "missing ${HARMFUL_CSV}" >&2; exit 2; }
  COMMON+=(
    --prompts_csv "${HARMFUL_CSV}" --prompt_column prompt
    --objective anti_openers --n_shards 8 --n_prompts 28 --max_prompt_tok 96
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

python -u "${ROOT}/research/refusal_vector/prompt_rewrite_opt.py" \
  "${COMMON[@]}" "${SEARCH[@]}" --out "${OUT}/${STEM}_s${S}.json"

echo "[done] ${ARM} shard ${S} $(date)"
