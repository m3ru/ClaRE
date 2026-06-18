#!/bin/bash
#SBATCH --job-name=test-claude-on-llama
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/test_claude_on_llama_%j.out
#SBATCH --error=logs/test_claude_on_llama_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Behavioral eval: feed each Claude rewrite to bare Llama-3-8B-Instruct (no
# adapter, no RWR system prompt) and capture its response. Regex-classify
# refused / compliant in-place. Also tests the original benign prompts as a
# control (one per benign_id).
#
# Expected: ~5,000 originals + 14,996 rewrites = ~20,000 generations.
# At batch 8 / greedy / 128 max tokens, roughly 1.5-2 h on H100 40GB.
#
# Skip the originals control: --no-include_originals
# Smoke test on N records first: --limit 100

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Starting Claude-rewrites-on-Llama behavioral eval at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -u test_claude_rewrites_on_llama.py \
    --jsonl  prompt_iteration_results/dataset_research_framing_full.jsonl \
    --output prompt_iteration_results/llama_behavioral_eval/llama_responses.jsonl \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --max_new_tokens 256 \
    --batch_size 8

echo "Behavioral eval generation completed at $(date)"
echo "scp the JSONL back to your laptop and run judge_with_claude.py to add"
echo "Claude-judge labels on the regex-compliant + sampled-refused records."
