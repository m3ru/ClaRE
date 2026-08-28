#!/bin/bash
#SBATCH -J or_gptoss
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 14:00:00
#SBATCH --output=$HOME/scratch/or_gptoss_%A_%a.out
#SBATCH --error=$HOME/scratch/or_gptoss_%A_%a.err
#SBATCH --array=0-31
#SBATCH --exclude=<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>,<compute-node>

set -euo pipefail
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

# ---- Array index → (dataset, shard_id, num_shards) ----
#   0-7:   cybermetric  (2000 prompts, 8 shards of ~250)
#   8-17:  sciq         (3000 prompts, 10 shards of ~300)
#  18-27:  medqa        (3000 prompts, 10 shards of ~300)
#  28-31:  pubmedqa     (1000 prompts, 4 shards of ~250)
IDX=${SLURM_ARRAY_TASK_ID}
if   (( IDX <= 7  )); then DATASET_NAME="cybermetric"; SHARD_ID=$((IDX));     NUM_SHARDS=8
elif (( IDX <= 17 )); then DATASET_NAME="sciq";        SHARD_ID=$((IDX-8));   NUM_SHARDS=10
elif (( IDX <= 27 )); then DATASET_NAME="medqa";       SHARD_ID=$((IDX-18));  NUM_SHARDS=10
elif (( IDX <= 31 )); then DATASET_NAME="pubmedqa";    SHARD_ID=$((IDX-28));  NUM_SHARDS=4
else echo "[error] Unknown array index ${IDX}"; exit 1
fi
echo "[job] Dataset: ${DATASET_NAME}, shard ${SHARD_ID}/${NUM_SHARDS} (array idx ${IDX})"

export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ] && [ -f "${HOME}/.cache/huggingface/token" ]; then
  export HUGGING_FACE_HUB_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi
export SCRATCH_BASE=${HOME}/scratch
export CONDA_PKGS_DIRS=${SCRATCH_BASE}/conda_pkgs
export CONDA_ENVS_PATH=${SCRATCH_BASE}/conda_envs
export PIP_CACHE_DIR=${SCRATCH_BASE}/pip_cache
export TMPDIR=${SCRATCH_BASE}/tmp
mkdir -p ${CONDA_PKGS_DIRS} ${CONDA_ENVS_PATH} ${PIP_CACHE_DIR} ${TMPDIR}
export HF_HOME=${HOME}/scratch/hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TORCH_HOME=${HOME}/scratch/torch_cache
export TRITON_CACHE_DIR=${HOME}/scratch/triton_cache
mkdir -p ${HF_HOME} ${HF_HUB_CACHE} ${HF_DATASETS_CACHE} ${TORCH_HOME} ${TRITON_CACHE_DIR}

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ~/scratch/conda_envs/or-para-env

echo "[env] Python: $(python --version)"
echo "[env] Node: $(hostname)"
nvidia-smi || true

# Verify GPU has >= 70GB VRAM (A100/H100/H200 only)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "[env] GPU memory: ${GPU_MEM} MiB"
if [ "${GPU_MEM}" -lt 70000 ]; then
  echo "[error] GPU too small (${GPU_MEM} MiB < 70000 MiB). Need A100/H100/H200." >&2
  exit 1
fi

SCRIPT_DIR=${HOME}/ppo_or
DATA_FILE=${HOME}/scratch/ppo_or_datasets/${DATASET_NAME}.jsonl
REFUSAL_VECTOR=${HOME}/scratch/ppo_or_data/refusal_vector.layer032.npz

if [ ! -f "${DATA_FILE}" ]; then
  python -u ${SCRIPT_DIR}/prepare_datasets.py --output_dir ${HOME}/scratch/ppo_or_datasets --dataset ${DATASET_NAME}
fi
echo "[step1] Dataset ready: $(wc -l < ${DATA_FILE}) prompts"

OUTDIR=${HOME}/scratch/ppo_or_data/or_gptoss_${DATASET_NAME}
mkdir -p ${OUTDIR}

python -u ${SCRIPT_DIR}/paraphrase_and_score.py \
  --data_path ${DATA_FILE} \
  --refusal_vector_path ${REFUSAL_VECTOR} \
  --output ${OUTDIR}/or_susceptibility_rankings.json \
  --gen_model openai/gpt-oss-120b \
  --scoring_model meta-llama/Meta-Llama-3-8B-Instruct \
  --max_prompts 3000 --n_paraphrases 20 --temperature 0.9 \
  --activation_layer 32 --score_batch_size 8 --checkpoint_every 25 \
  --gen_batch_size 5 --shard_id ${SHARD_ID} --num_shards ${NUM_SHARDS}

echo "[done] ${DATASET_NAME} shard ${SHARD_ID}/${NUM_SHARDS} complete"
