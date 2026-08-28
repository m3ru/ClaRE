#!/bin/bash
#SBATCH --job-name=eval-rwr-llama-tune
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/eval_rwr_llama_tune_%j.out
#SBATCH --error=logs/eval_rwr_llama_tune_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<EMAIL>
#SBATCH --export=ALL

# Behavioral eval of a logit tuning variant on the SAME held-out prompts as every
# prior arm. First checkpoint also runs the base-Llama-as-attacker arm (matched
# baseline), mirroring run_eval_rwr_llama.slurm's protocol.
# Usage: sbatch run_eval_rwr_llama_tune.slurm <tag> <ckpt1> [ckpt2 ...]
#   e.g.  sbatch run_eval_rwr_llama_tune.slurm _tune1_e6 final epoch_4
set -euo pipefail
TAG="${1:?usage: sbatch run_eval_rwr_llama_tune.slurm <tag> <ckpt1> [ckpt2 ...]}"
shift
[ "$#" -ge 1 ] || { echo "need at least one checkpoint subdir"; exit 1; }

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs "probe_or/results/eval_llama_logit${TAG}"

module load gcc arrow/23.0.1 cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CKPT="probe_or/results/rwr_llama_ckpt_logit${TAG}"
HELDOUT=probe_or/results/llama_scored/heldout_originals.json
FIRST=1
for SUB in "$@"; do
    BASEFLAG=""
    [ "$FIRST" = 1 ] && BASEFLAG="--eval_base_model"
    FIRST=0
    echo "Evaluating ${CKPT}/${SUB} at $(date)"
    python -u eval_rwr_llama.py \
        --adapter_dir "$CKPT/$SUB" --heldout_json "$HELDOUT" \
        --output "probe_or/results/eval_llama_logit${TAG}/eval_${SUB}.json" \
        --base_model meta-llama/Meta-Llama-3-8B-Instruct \
        --n_rewrites 4 --n_refusal_samples 4 --max_eval_prompts 200 $BASEFLAG
done
echo "Eval complete at $(date)"
