#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=reward-dirs-h2h
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Needs the ablation-selected direction from run_arditi_sweep_qwen.slurm, which saves
# arditi_direction.npz (d_hat at the selected layer + all_d_hat for every layer).
A_L=probe_or/results/arditi_sweep_llama/arditi_direction.npz
A_Q=probe_or/results/arditi_sweep_qwen/arditi_direction.npz
for f in "$A_L" "$A_Q"; do
  [ -f "$f" ] || { echo "MISSING $f -- run run_arditi_sweep_qwen.slurm first"; exit 1; }
done
LSTAR_L=$(python -c "import numpy as np;print(int(np.load('$A_L')['layer']))")
LSTAR_Q=$(python -c "import numpy as np;print(int(np.load('$A_Q')['layer']))")
AUC_L=$(python -c "import numpy as np;print(int(np.load('probe_or/results/llama_signals/probe_absolute.npz')['best_layer']))")
AUC_Q=$(python -c "import numpy as np;print(int(np.load('probe_or/results/qwen_signals/probe_absolute.npz')['best_layer']))")
echo "Llama: ablation L$LSTAR_L vs AUC L$AUC_L | Qwen: ablation L$LSTAR_Q vs AUC L$AUC_Q"

echo "### Llama"
python -u compare_reward_directions.py \
  --pool_csv probe_or/results/llama_scored/sonnet_benign_llama_or.csv \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dir_a "$A_L:$LSTAR_L:d_hat" \
  --dir_b "probe_or/results/llama_signals/probe_absolute.npz:$AUC_L:d" \
  --out probe_or/results/reward_dirs_h2h_llama.json --batch_size 16

echo "### Qwen"
python -u compare_reward_directions.py \
  --pool_csv probe_or/results/qwen_scored/sonnet_benign_qwen_3sig.csv \
  --base_model Qwen/Qwen3-32B \
  --dir_a "$A_Q:$LSTAR_Q:d_hat" \
  --dir_b "probe_or/results/qwen_signals/probe_absolute.npz:$AUC_Q:d" \
  --out probe_or/results/reward_dirs_h2h_qwen.json --batch_size 8
