#!/bin/bash
#SBATCH --job-name=qwen-causal
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/qwen_causal_%j.out
#SBATCH --error=logs/qwen_causal_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# Arditi causal test for Qwen3-32B: locate the causal refusal layer (ablation necessity +
# addition sufficiency) to set the Qwen vector layer, and check whether it differs from the
# behavioral-best L58 (as Llama's causal L17 differs from its behavioral-best L31).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs refusal_atlas/results

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Qwen3-32B causal refusal test at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# CLEAN re-run (fixes fable review #1,#2): L63 not L64 (hs[64] is post-final-RMSNorm), and
# --test_offset 2000 draws the harmful/harmless EVAL prompts DISJOINT from the direction-fit
# rows [0:2000], removing the necessity/sufficiency circularity of the first pass.
VX=../../refusal_vector/3_Vector_Extraction
python -u refusal_atlas/causal_refusal_qwen.py \
    --harmful_csv "$VX/final_refusals_prompts.csv" \
    --benign_csv  "$VX/final_benign_prompts.csv" \
    --refusal_csv "$VX/final_refusals_prompts.csv" \
    --out refusal_atlas/results/qwen_causal_results_clean.json \
    --base_model Qwen/Qwen3-32B \
    --layers 32,45,58,63 --add_coefs 1,2,4 --test_offset 2000 \
    --n_harmful 100 --n_harmless 100 --n_dir 2000 \
    --max_new_tokens 48 --batch_size 8

echo "Done at $(date)"
