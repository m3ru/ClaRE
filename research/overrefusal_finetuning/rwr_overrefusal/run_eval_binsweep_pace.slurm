#!/bin/bash
#SBATCH -J rwr_eval_binsweep
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Evaluate baseline + all bin-sweep models on the held-out alpaca set (eval_seed=99,
# disjoint from the 2500 training + 20 iteration prompts). Generates rewrites and
# scores per-generation similarity + refusal_delta (both k-independent). OR is
# recomputed offline afterward at k=18.4/c=0.5/d=100 (training scale) and k=5 (ref).
set -euo pipefail
echo "[stage] eval_binsweep on $(hostname) at $(date)"
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

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

RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"
cd "${RWR_DIR}"
SWEEP="${SCRATCH_BASE}/rwr_binsweep"
OUT_DIR="prompt_iteration_results/held_out_eval_binsweep"

MODELS=("baseline=baseline")
for TAG in keepall_uniform v1gentle top40_16x top20_only top30_decile top10_only; do
  if [ -d "${SWEEP}/${TAG}/final" ]; then MODELS+=("${TAG}=${SWEEP}/${TAG}/final");
  else echo "[warn] missing ${SWEEP}/${TAG}/final"; fi
done
echo "[stage] models: ${MODELS[*]}"

python -u eval_held_out.py \
  --models "${MODELS[@]}" \
  --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir "${OUT_DIR}" \
  --n_alpaca 200 \
  --n_dolly 0 \
  --n_per_prompt 3 \
  --temperature 0.7 \
  --max_new_tokens 64 \
  --gen_batch_size 4 \
  --score_batch_size 8 \
  --eval_seed 99 \
  --training_seed 42 \
  --training_n_alpaca 2500 \
  --iter_seed 42 \
  --iter_n 20

echo "[done] eval_binsweep complete at $(date)"
