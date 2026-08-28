#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=ur-gen
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/ur_gen_%j.out
#SBATCH --error=logs/ur_gen_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u gen_underrefusal.py --out_dir probe_or/results/underrefusal/gen \
    --max_seeds ${UR_SEEDS:-1100} --n_rewrites ${UR_REPS:-4}
