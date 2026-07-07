#!/bin/bash
#SBATCH -J layer_sweep_icannot
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Layer sweep: per-layer refusal-direction activation delta vs 'I cannot' logprob
# delta over the Claude rewrites. Reuses the icannot pairs CSV for the behavioral
# signal; rebuilds per-layer diff-of-means directions from the split subsample.
set -euo pipefail
echo "[stage] layer_sweep_icannot on $(hostname) at $(date)"
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
python -c "import torch,transformers,scipy" || { echo "[error] env broken"; exit 3; }

RWR_DIR="${HOME}/ClaRE/research/overrefusal_finetuning/rwr_overrefusal"
cd "${RWR_DIR}"
PAIRS="prompt_iteration_results/icannot_vs_refusal/icannot_vs_refusal_pairs.csv"
SPLIT="${SCRATCH_BASE}/split_samples"
[ -f "${PAIRS}" ] || { echo "[error] missing ${PAIRS}"; exit 2; }
[ -f "${SPLIT}/split_benign_sample.csv" ] || { echo "[error] missing split samples"; exit 2; }

python -u layer_sweep_icannot.py \
  --pairs_csv "${PAIRS}" \
  --benign_csv "${SPLIT}/split_benign_sample.csv" \
  --refusal_csv "${SPLIT}/split_refusal_sample.csv" \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --out_dir prompt_iteration_results/layer_sweep_icannot \
  --batch_size 16 \
  --sim_hi 0.85

echo "[done] layer_sweep_icannot complete at $(date)"
