#!/bin/bash
#SBATCH -J rwr_binsweep
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Bin-config strictness sweep. Re-run claude_rwr_fixed on the CORRECTED (FIXED)
# Claude data but with the ORIGINAL v1 scoring stats for binning:
#   k (similarity_exponent) = 18.4, c (similarity_center) = 0.5, d (refusal_divisor) = 100
# (confirmed from scratch/rwr_v1_dolly_checkpoints/rwr_config.json). Only the bin
# granularity/weights vary across the array, from keep-everything to top-10%-only.
set -euo pipefail
echo "[stage] rwr_binsweep idx=${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- config table (TAG NUM_BINS BIN_WEIGHTS) ----
case "${SLURM_ARRAY_TASK_ID}" in
  0) TAG=keepall_uniform; NUM_BINS=4;  W="1,1,1,1" ;;
  1) TAG=v1gentle;        NUM_BINS=5;  W="1,2,4,8,16" ;;
  2) TAG=top40_16x;       NUM_BINS=5;  W="0,0,0,1,16" ;;
  3) TAG=top20_only;      NUM_BINS=5;  W="0,0,0,0,1" ;;
  4) TAG=top30_decile;    NUM_BINS=10; W="0,0,0,0,0,0,0,1,8,32" ;;
  5) TAG=top10_only;      NUM_BINS=10; W="0,0,0,0,0,0,0,0,0,1" ;;
  *) echo "[error] unknown idx ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
echo "[stage] TAG=${TAG} num_bins=${NUM_BINS} bin_weights=${W}  (k=18.4 c=0.5 d=100)"

export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
python -c "import torch,transformers,peft,sentence_transformers" || { echo "[error] env broken"; exit 3; }
echo "[stage] python: $(python --version)"

RWR_DIR="${HOME}/anon-repo/research/overrefusal_finetuning/rwr_overrefusal"
SHARD_DIR="${SCRATCH_BASE}/rwr_claude_data_fixed"
OUT_DIR="${SCRATCH_BASE}/rwr_binsweep/${TAG}"
[ -d "${SHARD_DIR}" ] || { echo "[error] missing shards ${SHARD_DIR}"; exit 2; }
mkdir -p "${OUT_DIR}"
cd "${RWR_DIR}"

python -u train_rwr.py \
  --shard_dir "${SHARD_DIR}" \
  --output_dir "${OUT_DIR}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1.5e-5 \
  --max_seq_length 512 \
  --num_bins "${NUM_BINS}" \
  --bin_weights "${W}" \
  --similarity_exponent 18.4 \
  --similarity_center 0.5 \
  --refusal_divisor 100.0 \
  --similarity_floor 0.5 \
  --seed 42 \
  --gradient_checkpointing

echo "[done] ${TAG} complete at $(date)"; ls -la "${OUT_DIR}"/final 2>&1 || true
