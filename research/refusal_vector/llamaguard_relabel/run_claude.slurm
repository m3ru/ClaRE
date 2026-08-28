#!/bin/bash
#SBATCH -J lg_claude
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 01:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_HOME="${HOME}/scratch/hf_cache" HF_HUB_CACHE="${HOME}/scratch/hf_cache/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_XET_DISABLE=1
export TMPDIR="${HOME}/tmp_relabel"; mkdir -p "${TMPDIR}"
module load anaconda3; source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${HOME}/scratch/conda_envs/ppo-or-env"
D="${HOME}/anon-repo/research/refusal_vector/llamaguard_relabel"
L3=""
for c in Vector_Extraction 3_Vector_Extraction; do
  p="${HOME}/anon-repo/research/refusal_vector/${c}/refusal_vector.layer032.npz"
  if [ -f "$p" ]; then L3="$p"; break; fi
done
[ -n "$L3" ] || { echo "[error] no llama3 vector"; exit 2; }
echo "llama3 vector: ${L3}"
cd "${D}"
python -u claude_score.py --claude "${D}/claude_heldout_alpaca.json" \
  --llama3_vector "${L3}" \
  --lg_vector "${D}/refusal_vector_llamaguard_relabeled.layer030.npz"
