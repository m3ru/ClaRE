#!/bin/bash
#SBATCH --job-name=rwr-qwen-probe
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/rwr_qwen_probe_%j.out
#SBATCH --error=logs/rwr_qwen_probe_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

# RWR-Qwen on the raw vector-projection-delta reward @ causal L58, k=18.4. Mirrors
# run_rwr_llama_vector.slurm. Quantile bin edges (q35/q65/q85, weights 0,1,4,16) set from
# the build_llama_shards "candidate --bin_edges" line for the PROBE arm in the scoring log
# (logs/score_qwen_3sig_*.out). Data-derived -> MUST be filled before submit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
BIN_EDGES="0.108,0.5485,1.956"   # probe q35/q65/q85 from job 19020842 shard-build log
[ -n "$BIN_EDGES" ] || { echo "ERROR: set BIN_EDGES from the score_qwen_3sig log (probe arm) first"; exit 1; }

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "RWR-Qwen (standardized probe (=vector) @ L58, k=18.4, QLoRA) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -u train_rwr.py \
    --base_model Qwen/Qwen3-32B --load_in_4bit \
    --shard_dir probe_or/results/qwen_scored/llama_shards_probe \
    --output_dir probe_or/results/rwr_qwen_ckpt_probe \
    --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --learning_rate 1.5e-5 \
    --max_seq_length 512 \
    --num_bins 4 --bin_edges "$BIN_EDGES" --bin_weights 0,1,4,16 \
    --similarity_exponent 18.4 --similarity_center 0.75 --refusal_divisor 1.0 \
    --similarity_floor 0.5 --num_samples_per_epoch 4000 --seed 42

echo "Training complete at $(date)"
