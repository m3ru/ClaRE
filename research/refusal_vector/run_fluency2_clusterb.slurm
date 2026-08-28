#!/bin/bash
#SBATCH -J flu2
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
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"; export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
DIR="${HOME}/anon-repo/research/refusal_vector"; cd "${DIR}"
OUT="${SCRATCH_BASE}/prompt_opt"; PAIRS="${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal/probe_or"

# ablation of the three new fluency knobs, all on the icannot objective
WW=""; EX=0; MX=0.0
case "${SLURM_ARRAY_TASK_ID}" in
  0) WW="--whole_word_only"; EX=0; MX=0.5; TAG=ww_mx ;;
  1) WW="--whole_word_only"; EX=6; MX=0.5; TAG=ww_mx_ex6 ;;
  2) WW="--whole_word_only"; EX=0; MX=0.0; TAG=ww_only ;;
  3) WW="";                  EX=6; MX=0.5; TAG=mx_ex6 ;;
  *) echo bad; exit 2 ;;
esac
echo "[cfg] ${TAG}: ww='${WW}' extra=${EX} nllmax=${MX}"

python -u prompt_rewrite_opt.py \
  --pairs_csv "${PAIRS}/pairs.csv" "${PAIRS}/pairs_sonnet.csv" \
  --dirs_npz "${OUT}/harmful_dirs_all_layers.npz" \
  --out "${OUT}/flu2_${TAG}.json" \
  --objective icannot --layer 12 \
  --n_prompts 60 --steps 150 --topk 256 --n_cand 96 \
  --sim_floor 0.85 --lam 20.0 --block_profanity \
  --fluency_weight 1.0 --lm_topk_filter 512 \
  --nll_max_weight ${MX} --extra_tokens ${EX} ${WW}

python -u score_llamaguard_flags.py \
  --in_json "${OUT}/flu2_${TAG}.json" --out "${OUT}/flu2_${TAG}_flagged.json"
echo "[done] ${TAG} $(date)"
