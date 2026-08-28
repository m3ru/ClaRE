#!/bin/bash
#SBATCH -J score_probe_or
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 02:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
echo "[stage] score_probe_or on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"; cd "${RWR_DIR}"
python -c "import torch,transformers,scipy" || { echo "[error] env"; exit 3; }
python -u score_probe_or.py \
  --benign_csv "${SCRATCH_BASE}/split_samples/split_benign_sample.csv" \
  --refusal_csv "${SCRATCH_BASE}/split_samples/split_refusal_sample.csv" \
  --pairs_csv probe_or/pairs.csv probe_or/pairs_sonnet.csv \
  --out_shards "${SCRATCH_BASE}/probe_or_shards" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct --batch_size 16
echo "[done] score_probe_or at $(date); shards: $(ls ${SCRATCH_BASE}/probe_or_shards/*.json 2>/dev/null | wc -l)"
