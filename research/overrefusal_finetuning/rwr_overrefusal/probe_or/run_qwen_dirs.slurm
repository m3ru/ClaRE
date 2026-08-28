#!/bin/bash
#SBATCH --job-name=qwen-dirs
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/qwen_dirs_%j.out
#SBATCH --error=logs/qwen_dirs_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir
mkdir -p logs probe_or/qwen_activations

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# NOTE: no HF_TOKEN -- the stored token is corrupt; Qwen3-32B is public & cached, offline load needs none.

echo "Qwen3-32B refusal/benign direction extraction at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# refuse/benign split only (no --pool_csvs) -> per-layer mass-mean directions, all 65 layers.
# batch 4: 32B (~65GB) on one 80GB H100 leaves little headroom; keep activations small.
python -u probe_or/extract_layer_acts.py \
    --refusal_csv ../../refusal_vector/3_Vector_Extraction/final_refusals_prompts.csv \
    --benign_csv  ../../refusal_vector/3_Vector_Extraction/final_benign_prompts.csv \
    --out_dir probe_or/qwen_activations \
    --base_model Qwen/Qwen3-32B \
    --n_per_class 2500 --batch_size 4 --max_length 1152 --acts_dtype float32

echo "Done at $(date)"
echo "Next: probe_or/probe_qwen_signals.py (after pair-acts + behavioral gen)"
