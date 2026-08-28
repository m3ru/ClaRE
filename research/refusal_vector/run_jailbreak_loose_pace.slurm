#!/bin/bash
#SBATCH -J jb_loose
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
export JB_SCRATCH="${HOME}/scratch"
export HF_HOME="${JB_SCRATCH}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"
mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${JB_SCRATCH}/conda_envs/ppo-or-env"

DIR="${HOME}/ClaRE/research/refusal_vector"
OUT="${JB_SCRATCH}/prompt_opt/jailbreak_flip_v1/loose"
DATA="${JB_SCRATCH}/prompt_opt/jailbreak_flip_v1/data/strongreject_custom_direct.csv"
S=${SLURM_ARRAY_TASK_ID}
cd "${DIR}"
mkdir -p "${OUT}"

# Inverted version of the original 12-slot scale configuration.
python -u prompt_rewrite_opt.py \
  --prompts_csv "${DATA}" --prompt_column prompt \
  --dirs_npz "${JB_SCRATCH}/prompt_opt/harmful_dirs_all_layers.npz" \
  --out "${OUT}/jb_loose_s${S}.json" \
  --objective anti_icannot --layer 12 \
  --shard_id "${S}" --n_shards 16 --n_prompts 20 --max_prompt_tok 96 \
  --steps 150 --topk 256 --n_cand 96 \
  --sim_floor 0.85 --lam 20.0 --block_profanity \
  --fluency_weight 1.0 --lm_topk_filter 512 \
  --nll_max_weight 0.5 --extra_tokens 12 \
  --whole_word_only --max_new_tokens 128

python -u score_llamaguard_flags.py \
  --in_json "${OUT}/jb_loose_s${S}.json" \
  --out "${OUT}/jb_loose_s${S}_flagged.json"
echo "[done] loose shard ${S} $(date)"
