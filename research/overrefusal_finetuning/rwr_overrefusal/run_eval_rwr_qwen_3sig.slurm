#!/bin/bash
#SBATCH --job-name=eval-rwr-qwen
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-10:00:00
#SBATCH --output=logs/eval_rwr_qwen_3sig_%j.out
#SBATCH --error=logs/eval_rwr_qwen_3sig_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

# Behavioral eval of one Qwen 3-signal RWR attacker on the COMMON held-out set.
# Uses eval_rwr_llama.py (fully --base_model parameterized, same model-agnostic classify,
# --heldout_json common-split design) with the Qwen base -- NOT eval_rwr_qwen.py, which
# derives its own split from --shard_dir (would not be common across arms).
# Full 80GB H100: bf16 32B ~64GB, MIG 40GB slice would OOM.
# Usage: sbatch run_eval_rwr_qwen_3sig.slurm <vector|probe|logit>
set -euo pipefail
SIGNAL="${1:?usage: sbatch run_eval_rwr_qwen_3sig.slurm <vector|probe|logit>}"
case "$SIGNAL" in vector|probe|logit) ;; *) echo "bad signal: $SIGNAL"; exit 1;; esac

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "probe_or/results/eval_qwen_3sig_${SIGNAL}"

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Behavioral eval of RWR-Qwen (${SIGNAL} reward) at $(date)"
CKPT="probe_or/results/rwr_qwen_ckpt_${SIGNAL}"
HELDOUT=probe_or/results/qwen_scored/heldout_originals.json

python -u eval_rwr_llama.py \
    --adapter_dir "$CKPT/final" --heldout_json "$HELDOUT" \
    --output "probe_or/results/eval_qwen_3sig_${SIGNAL}/eval_final.json" \
    --base_model Qwen/Qwen3-32B \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200 --eval_base_model

python -u eval_rwr_llama.py \
    --adapter_dir "$CKPT/epoch_2" --heldout_json "$HELDOUT" \
    --output "probe_or/results/eval_qwen_3sig_${SIGNAL}/eval_epoch2.json" \
    --base_model Qwen/Qwen3-32B \
    --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200

echo "Eval complete at $(date)"
