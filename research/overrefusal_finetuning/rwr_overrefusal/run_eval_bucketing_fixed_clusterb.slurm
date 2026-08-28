#!/bin/bash
#SBATCH -J rwr_eval_bucketing_fixed
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Head-to-head (PDF eval a): OR head-to-head with the bug-free scorer on the
# published held-out set (eval_seed=99, 200 alpaca + 200 dolly x 3 gens, disjoint
# from training/prompt-iteration). Compares:
#   baseline  vs  claude_rwr_fixed (Exp1, bins5)  vs  claude_rwr_v1_fixed (Exp2, bins10)
# Reports mean + p90/p95 tail. Then re-scores at k=5.0 for apples-to-apples with
# the published claude_rwr numbers in claude_training_results.md.
set -euo pipefail
echo "[stage] eval_bucketing_fixed on $(hostname) at $(date)"
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
echo "[stage] python: $(python --version)"; nvidia-smi | head -12 || true

RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"
cd "${RWR_DIR}"

EXP1="${SCRATCH_BASE}/rwr_claude_fixed_checkpoints/final"
V1="${SCRATCH_BASE}/rwr_claude_v1_fixed_checkpoints/final"
OUT_DIR="prompt_iteration_results/held_out_eval_bucketing_fixed"

MODELS=("baseline=baseline")
[ -d "${EXP1}" ] && MODELS+=("claude_rwr_fixed=${EXP1}")   || echo "[warn] missing ${EXP1}"
[ -d "${V1}"   ] && MODELS+=("claude_rwr_v1_fixed=${V1}")  || echo "[warn] missing ${V1}"
echo "[stage] models: ${MODELS[*]}"

python -u eval_held_out.py \
  --models "${MODELS[@]}" \
  --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dolly_csv ../../overrefusal_sampling/data/benign_dolly.csv \
  --output_dir "${OUT_DIR}" \
  --n_alpaca 200 \
  --n_dolly 200 \
  --n_per_prompt 3 \
  --temperature 0.7 \
  --max_new_tokens 64 \
  --gen_batch_size 4 \
  --score_batch_size 8 \
  --eval_seed 99 \
  --training_seed 42 \
  --training_n_alpaca 2500 \
  --training_n_dolly 2500 \
  --iter_seed 42 \
  --iter_n 20

echo "[stage] re-score at k=5.0 for comparability with published numbers"
python -u rescore_held_out.py \
  --input  "${OUT_DIR}/held_out_eval_results.json" \
  --output "${OUT_DIR}/held_out_eval_results_k5.json"

echo "[done] eval_bucketing_fixed complete at $(date)"
