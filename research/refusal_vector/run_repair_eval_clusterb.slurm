#!/bin/bash
#SBATCH -J repair_eval
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export REPAIR_SCRATCH="${HOME}/scratch"
export HF_HOME="${REPAIR_SCRATCH}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_repair_eval"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${REPAIR_SCRATCH}/conda_envs/ppo-or-env"
DIR="${HOME}/ClaRE/research/refusal_vector"; cd "${DIR}"
IN="${REPAIR_SCRATCH}/prompt_opt/repair/grammatical_repairs.json"
OUT="${REPAIR_SCRATCH}/prompt_opt/repair/eval"; mkdir -p "${OUT}"
test -s "${IN}"
python -u evaluate_grammatical_repairs.py \
  --input "${IN}" --out "${OUT}/repair_eval_s${SLURM_ARRAY_TASK_ID}.json" \
  --shard_id "${SLURM_ARRAY_TASK_ID}" --n_shards 4 \
  --batch_size 8 --max_new_tokens 128 --with_llamaguard
