#!/bin/bash
#SBATCH -J rewrite_opt
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
echo "[stage] rewrite_opt cfg=${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"; export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
DIR="${HOME}/ClaRE/research/refusal_vector"; cd "${DIR}"
OUT="${SCRATCH_BASE}/prompt_opt"; mkdir -p "${OUT}"
PAIRS="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal/probe_or"

# FLU=1 turns on the fluency penalty + LM-constrained proposals
FW=0; LMK=0; FSUF=""
case "${SLURM_ARRAY_TASK_ID}" in
  0) OBJ=icannot;   L=12 ;;
  1) OBJ=direction; L=12 ;;
  2) OBJ=direction; L=17 ;;
  3) OBJ=icannot;   L=12; FW=4.0; LMK=128; FSUF="_hard" ;;
  *) echo "bad cfg"; exit 2 ;;
esac
if [ "${FLU:-0}" = "1" ]; then
  [ "${FW}" = "0" ] && { FW=1.0; LMK=512; }
  FSUF="_flu${FSUF}"
else
  FW=0; LMK=0; FSUF=""
fi
# BLOCK=1 forbids taboo/harm-adjacent tokens, forcing NON-lexical over-refusal
BLOCK="${BLOCK:-0}"
EXTRA=""; TAGSUF=""
if [ "${BLOCK}" = "1" ]; then EXTRA="--block_profanity"; TAGSUF="_noTaboo"; fi
TAG="rw_${OBJ}_L${L}${TAGSUF}${FSUF}"
echo "[cfg] ${TAG}"

python -u prompt_rewrite_opt.py \
  --pairs_csv "${PAIRS}/pairs.csv" "${PAIRS}/pairs_sonnet.csv" \
  --dirs_npz "${OUT}/harmful_dirs_all_layers.npz" \
  --out "${OUT}/${TAG}.json" \
  --objective "${OBJ}" --layer "${L}" \
  --n_prompts 60 --steps 150 --topk 256 --n_cand 96 \
  --sim_floor 0.85 --lam 20.0 ${EXTRA} \
  --fluency_weight ${FW} --lm_topk_filter ${LMK}

python -u score_llamaguard_flags.py \
  --in_json "${OUT}/${TAG}.json" --out "${OUT}/${TAG}_flagged.json"
echo "[done] ${TAG} at $(date)"
