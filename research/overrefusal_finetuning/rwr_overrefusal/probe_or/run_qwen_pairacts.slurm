#!/bin/bash
#SBATCH --job-name=qwen-pairacts
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/qwen_pairacts_%j.out
#SBATCH --error=logs/qwen_pairacts_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/qwen_pair_acts

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Qwen3-32B pair-acts (orig+rewrite, all layers) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Row order aligned with pairs_qwen.csv -> idx-aligned with gen_qwen_refusal output.
python -u probe_or/extract_pair_acts.py \
    --pairs_csv probe_or/pairs_qwen.csv \
    --out_dir probe_or/qwen_pair_acts \
    --base_model Qwen/Qwen3-32B \
    --batch_size 4 --max_length 512 --acts_dtype float32

echo "Done at $(date)"
