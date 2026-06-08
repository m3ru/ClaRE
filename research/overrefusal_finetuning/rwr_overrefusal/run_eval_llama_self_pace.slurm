#!/bin/bash
#SBATCH --job-name=llama-self-eval
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/llama_self_eval_%j.out
#SBATCH --error=logs/llama_self_eval_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export HF_HOME="${HOME}/scratch/hf_cache"; export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_DATASETS_OFFLINE=1
PY=~/scratch/conda_envs/ppo-or-env/bin/python
echo "start $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader || true
$PY -u eval_held_out.py \
  --models llama_self_rwr=$HOME/scratch/rwr_llama_self_checkpoints/final \
  --refusal_vector_path ../../refusal_vector/Vector_Extraction/refusal_vector.layer032.npz \
  --dolly_csv ../../overrefusal_sampling/data/benign_dolly.csv \
  --output_dir prompt_iteration_results/held_out_eval_llama_self \
  --n_alpaca 200 --n_dolly 200 --n_per_prompt 3
echo "done $(date)"
