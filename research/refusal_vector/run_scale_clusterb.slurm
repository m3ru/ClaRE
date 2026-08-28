#!/bin/bash
#SBATCH -J or_scale
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-15
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
OUT="${SCRATCH_BASE}/prompt_opt/scale"; mkdir -p "${OUT}"
PAIRS="${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal/probe_or"
S=${SLURM_ARRAY_TASK_ID}

# Winning config from the flu2 ablation: whole-word + extra slots + worst-token NLL.
# --shard_id/--n_shards make the 16 tasks cover DISJOINT prompts (same seed everywhere,
# so the permutation and the held-out tail are identical across tasks by construction).
python -u prompt_rewrite_opt.py \
  --pairs_csv "${PAIRS}/pairs.csv" "${PAIRS}/pairs_sonnet.csv" \
  --dirs_npz "${SCRATCH_BASE}/prompt_opt/harmful_dirs_all_layers.npz" \
  --out "${OUT}/scale_s${S}.json" \
  --objective icannot --layer 12 \
  --shard_id ${S} --n_shards 16 --holdout_frac 0.30 \
  --dump_holdout "${OUT}/holdout_originals.json" \
  --n_prompts 200 --steps 150 --topk 256 --n_cand 96 \
  --sim_floor 0.85 --lam 20.0 --block_profanity \
  --fluency_weight 1.0 --lm_topk_filter 512 \
  --nll_max_weight 0.5 --extra_tokens 12 \
  --whole_word_only --max_new_tokens 128

python -u score_llamaguard_flags.py \
  --in_json "${OUT}/scale_s${S}.json" --out "${OUT}/scale_s${S}_flagged.json"
echo "[done] shard ${S} $(date)"
