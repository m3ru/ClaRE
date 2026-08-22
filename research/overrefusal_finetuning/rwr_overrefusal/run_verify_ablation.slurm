#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --job-name=verify-abl
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-02:30:00
#SBATCH --output=logs/verify_abl_%j.out
#SBATCH --error=logs/verify_abl_%j.err
#SBATCH --export=ALL
# Re-measure the single-direction ablations KEEPING the response text, so the harmful column
# can be judged instead of regex-matched. Same directions, same layer, same eval sets as the
# runs that produced the numbers in the tables. Both models: Llama is the control whose
# directions are NOT output-aligned, so its numbers should reproduce.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -c "import torch, transformers, numpy; print('[pre-flight] imports OK')"

echo "### Qwen3-32B  $(date)"
python -u verify_ablation_harmful.py --base_model Qwen/Qwen3-32B \
  --dirs_npz probe_or/results/delta_qwen/qwen_own_frame_dirs.npz \
  --atlas probe_or/results/qwen_signals/probe_absolute.npz \
  --n_eval 400 --n_harmful 200 --n_random 3 --batch_size 8 \
  --out_dir probe_or/results/verify_abl_qwen

echo "### Llama-3-8B control  $(date)"
python -u verify_ablation_harmful.py --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dirs_npz probe_or/results/delta/sym_frame_dirs.npz \
  --atlas probe_or/results/llama_signals/probe_absolute.npz \
  --n_eval 400 --n_harmful 200 --n_random 3 --batch_size 16 \
  --out_dir probe_or/results/verify_abl_llama
echo "### done $(date)"
