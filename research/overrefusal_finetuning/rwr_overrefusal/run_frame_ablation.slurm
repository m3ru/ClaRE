#!/bin/bash
#SBATCH --account=def-vganesh
#SBATCH --job-name=frame-abl
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=logs/killswitch_%j.out
#SBATCH --error=logs/killswitch_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda scipy-stack/2024b gcc arrow/23.0.1
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u ablate_frames.py --n_per_frame 400 --n_harmful 200 --out probe_or/results/frame_ablation_n400.json
