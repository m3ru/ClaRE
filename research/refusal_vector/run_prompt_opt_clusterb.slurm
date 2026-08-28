#!/bin/bash
#SBATCH -J prompt_opt
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail
echo "[stage] prompt_opt cfg=${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export SCRATCH_BASE="${HOME}/scratch"; export HF_HOME="${SCRATCH_BASE}/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SCRATCH_BASE}/conda_envs/ppo-or-env"
DIR="${HOME}/anon-repo/research/refusal_vector"; cd "${DIR}"
OUT="${SCRATCH_BASE}/prompt_opt"; mkdir -p "${OUT}"

# cfg: objective / method / layer
# L12 is the ABLITERATION-selected layer (min ablation-bypass) -- the chosen vector, so it
# runs first. L17 (best behavioral readout) trails as the comparison arm: whether an
# input-space attack works better on the causal lever or the readout is the open question.
case "${SLURM_ARRAY_TASK_ID}" in
  0) OBJ=direction; METH=gcg; L=12 ;;
  1) OBJ=direction; METH=pez; L=12 ;;
  2) OBJ=icannot;   METH=gcg; L=12 ;;
  3) OBJ=icannot;   METH=pez; L=12 ;;
  4) OBJ=direction; METH=gcg; L=17 ;;
  5) OBJ=direction; METH=pez; L=17 ;;
  *) echo "bad cfg"; exit 2 ;;
esac
# BLOCK=1 forbids profanity/violence tokens -> any refusal is refusal of benign-LOOKING text
BLOCK="${BLOCK:-0}"
EXTRA=""; TAGSUF=""
if [ "${BLOCK}" = "1" ]; then EXTRA="--block_profanity"; TAGSUF="_clean"; fi
TAG="${OBJ}_${METH}_L${L}${TAGSUF}"
echo "[cfg] objective=${OBJ} method=${METH} layer=${L} -> ${TAG}"

python -u prompt_opt.py \
  --dirs_npz "${OUT}/harmful_dirs_all_layers.npz" \
  --out "${OUT}/prompt_opt_${TAG}.json" \
  --objective "${OBJ}" --method "${METH}" --layer "${L}" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --n_adv 20 --n_train 8 --n_eval 60 --steps 250 \
  --topk 256 --n_cand 96 --cand_prompts 4 --n_random 3 ${EXTRA}
echo "[done] prompt_opt ${TAG} at $(date)"
