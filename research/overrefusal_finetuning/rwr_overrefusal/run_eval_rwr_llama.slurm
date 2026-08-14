#!/bin/bash
#SBATCH --job-name=eval-rwr-llama
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/eval_rwr_llama_%j.out
#SBATCH --error=logs/eval_rwr_llama_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Behavioral eval of one corrected-signal Llama RWR attacker.
# Usage: sbatch run_eval_rwr_llama.slurm <vector|probe|logit> [tag]
#   tag (optional) selects a layer variant, e.g. `vector _L31` evals
#   rwr_llama_ckpt_vector_L31 -> eval_llama_vector_L31. Same held-out set either way,
#   so every arm (incl. the L31 variant) is directly comparable.
set -euo pipefail
SIGNAL="${1:?usage: sbatch run_eval_rwr_llama.slurm <vector|probe|logit> [tag]}"
case "$SIGNAL" in vector|probe|logit) ;; *) echo "bad signal: $SIGNAL"; exit 1;; esac
TAG="${2:-}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "probe_or/results/eval_llama_${SIGNAL}${TAG}"

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Behavioral eval of RWR-Llama (${SIGNAL}${TAG} reward) at $(date)"
CKPT="probe_or/results/rwr_llama_ckpt_${SIGNAL}${TAG}"
HELDOUT=probe_or/results/llama_scored/heldout_originals.json

python -u eval_rwr_llama.py \
    --adapter_dir "$CKPT/final" --heldout_json "$HELDOUT" \
    --output "probe_or/results/eval_llama_${SIGNAL}${TAG}/eval_final.json" \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200 --eval_base_model

python -u eval_rwr_llama.py \
    --adapter_dir "$CKPT/epoch_2" --heldout_json "$HELDOUT" \
    --output "probe_or/results/eval_llama_${SIGNAL}${TAG}/eval_epoch2.json" \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200

echo "Eval complete at $(date)"
