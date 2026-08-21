#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --job-name=qwen-causal
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=logs/qwen_causal_%j.out
#SBATCH --error=logs/qwen_causal_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
Q=Qwen/Qwen3-32B
A=probe_or/results/qwen_signals/probe_absolute.npz
D=probe_or/results/delta_qwen/causal_dirs.npz

echo "### Qwen causal rank sweep (mirror of the Llama run)"
python -u ablate_causal_rank.py --base_model "$Q" --atlas "$A" --dirs_npz "$D" \
    --n_eval 400 --n_harmful 200 --batch_size 8 \
    --out probe_or/results/causal_rank_qwen.json

echo "### Qwen powered safety test"
python -u safety_power_check.py --base_model "$Q" --atlas "$A" --dirs_npz "$D" \
    --n_or 400 --batch_size 8 \
    --out probe_or/results/safety_power_qwen.json
