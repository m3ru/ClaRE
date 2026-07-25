#!/bin/bash
#SBATCH --job-name=qwen-score
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/qwen_score_%j.out
#SBATCH --error=logs/qwen_score_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgopalan6@gatech.edu
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs probe_or/results/qwen_scored

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "Score Sonnet benign pool with Qwen3-32B delta-probe (L58) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ~41.8k unique texts -> ~10 min + load. Uses the fitted scorer qwen_probe_raw.npz (= L58).
python -u probe_or/score_qwen_or.py \
    --pairs_csv logprob_or/results/icannot_or_sonnet/icannot_or_pairs_benign.csv \
    --scorer probe_or/results/qwen_signals/qwen_probe_raw.npz \
    --out probe_or/results/qwen_scored/sonnet_benign_qwen_or.csv \
    --base_model Qwen/Qwen3-32B --k 18.4 --c 0.75 --batch_size 8 --max_length 512

echo "Done at $(date)"
