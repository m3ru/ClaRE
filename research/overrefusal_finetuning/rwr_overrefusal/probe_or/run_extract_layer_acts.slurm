#!/bin/bash
#SBATCH --job-name=probe-extract
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/probe_extract_%j.out
#SBATCH --error=logs/probe_extract_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"          # submit from the rwr_overrefusal dir (parent of probe_or/)
mkdir -p logs probe_or/activations

module load cuda scipy-stack/2024b
source ~/general/bin/activate

export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_TOKEN="$(cat "$HF_HOME/token" 2>/dev/null || true)"

echo "Extracting per-layer activations at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Route (b): directions from the INDEPENDENT refusal/benign split that built the
# original refusal vector. Eval pool = the icannot-OR scored Claude+orp3k texts.
python -u probe_or/extract_layer_acts.py \
    --refusal_csv ../../refusal_vector/3_Vector_Extraction/final_refusals_prompts.csv \
    --benign_csv  ../../refusal_vector/3_Vector_Extraction/final_benign_prompts.csv \
    --pool_csvs logprob_or/results/icannot_or_claude/icannot_or_pairs.csv \
                logprob_or/results/icannot_or_orp3k/icannot_or_pairs.csv \
    --out_dir probe_or/activations \
    --n_per_class 2500 --max_eval 12000 --batch_size 16

echo "Extraction complete at $(date)"
echo "Next (login node): source ~/general/bin/activate && python probe_or/probe_ensemble.py --acts_dir probe_or/activations"
