#!/bin/bash
# Score INDUCED over-refusal on Qwen3-32B. Independent of the Llama job. Needs a full H100
# (65 GB of bf16 weights will not fit the 40GB MIG slice), so it may sit in ReqNodeNotAvail
# while H100 nodes are down -- the Llama arm is a complete result on its own.
#
# The same 250 originals x 8 arms (unmodified + Qwen's 5 frames + ctrl_rare + ctrl_common)
# = ~2,000 greedy generations at 48 new tokens.
#
# Wall-clock: the ownframes run on a full H100 did 400 prompts / 115 s at batch 8 = 0.287
# s/prompt -> ~10 min at batch 8, ~6 min at batch 16 (32B decode is weight-bandwidth-bound,
# so a bigger batch costs almost nothing per step). score_induced.py halves the batch and
# retries on OOM, so batch 16 cannot kill the job. 1:00 requested to cover model load.
#
# Qwen scores ONLY the five frames mined from Qwen's own low-edit triggers
# (probe_or/results/delta_qwen/qwen_frames.json) -- including fabrication, excluding
# Llama's intrusion.
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=induced-qwen
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --output=logs/induced_qwen_%j.out
#SBATCH --error=logs/induced_qwen_%j.err

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "score induced qwen start $(date)"
python -u score_induced.py --model qwen \
  --inserted probe_or/results/induction/inserted_sonnet.csv \
  --max_new_tokens 48 --batch_size 16 \
  --out probe_or/results/induction/induced_qwen.json
echo "score induced qwen done $(date)"
