#!/bin/bash
# Score INDUCED over-refusal on Llama-3-8B-Instruct. Self-contained; no dependency on the
# Qwen job. Plain generation, no ablation hooks, so the 40GB MIG slice is ample.
#
# 250 fresh Alpaca originals (seed-43 pool, offset 46000+, disjoint from every prior job) x
# 8 arms (unmodified + Llama's 5 frames + ctrl_rare + ctrl_common) = ~2,000 greedy
# generations at 48 new tokens.
#
# Wall-clock: the symsearch run on this same MIG slice did 400 prompts / 52 s at these exact
# settings = 0.13 s/prompt -> ~4.5 min, plus ~2 min model load. 0:40 requested.
#
# Llama scores ONLY the five frames mined from Llama's own low-edit triggers
# (analyze_frames.FRAMES). Qwen's fabrication arm is present in the inserted CSV and is
# deliberately skipped here.
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=induced-llama
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=logs/induced_llama_%j.out
#SBATCH --error=logs/induced_llama_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "score induced llama start $(date)"
python -u score_induced.py --model llama \
  --inserted probe_or/results/induction/inserted_sonnet.csv \
  --max_new_tokens 48 --batch_size 8 \
  --out probe_or/results/induction/induced_llama.json
echo "score induced llama done $(date)"
