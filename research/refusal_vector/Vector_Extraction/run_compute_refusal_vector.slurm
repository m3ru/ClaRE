#!/bin/bash
#SBATCH -J compute_refusal_vec
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENVNAME=llama8b-env
conda activate "${ENVNAME}"

PY="python"

BENIGN_DIR="${HOME}/scratch/acts_benign"
REFUSAL_DIR="${HOME}/scratch/acts_refusal"
OUT="${HOME}/scratch/refusal_vector.npz"

SELECT_LAYER=${SELECT_LAYER:-max}

${PY} -u "${HOME}/compute_refusal_vector.py" \
  --benign_dir "${BENIGN_DIR}" \
  --refusal_dir "${REFUSAL_DIR}" \
  --output "${OUT}" \
  --select_layer "${SELECT_LAYER}"

echo "Refusal vector saved to ${OUT}"


