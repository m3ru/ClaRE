#!/bin/bash
#SBATCH --job-name=llama-behav
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/llama_behav_%j.out
#SBATCH --error=logs/llama_behav_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

# Broad-behavioral refusal labels for the Llama probe REFIT (fixes the biased single-phrase
# p_icannot label: 40/6000 induced -> broad classifier recovers "I apologize"/"Sorry"/etc).
# Samples Llama on ALL 6000 pairs, row-aligned to probe_or/pair_acts (pair_meta), so the
# output idx indexes directly into acts_orig/acts_rw for probe_qwen_signals.py.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/llama_behav

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Llama-3-8B broad behavioral labeling (probe refit) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u probe_or/gen_qwen_refusal.py \
    --pairs_csv probe_or/pairs.csv \
    --out probe_or/results/llama_behav/behav.csv \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct \
    --n_samples 4 --temperature 0.7 --max_new_tokens 64 --batch_size 16 --seed 42

echo "Done at $(date)"
