#!/bin/bash
# Causal injection test: generate responses to the injected prompts.
#   sbatch --export=ALL,TARGET=llama run_injections.slurm
#   sbatch --export=ALL,TARGET=qwen  run_injections.slurm
#
# Walltime is deliberately short so the job stays backfill-eligible: fair-share on this
# account is exhausted, and a right-sized ask is the difference between minutes and days
# in the queue. Llama-3-8B needs well under an hour for 3,000 short generations; Qwen3-32B
# is 65 GB of weights but plain generation is far cheaper than the GCG search was.
#SBATCH --job-name=inject
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/inject_%j.out
#SBATCH --error=logs/inject_%j.err

set -euo pipefail
: "${TARGET:?submit with --export=ALL,TARGET=llama|qwen}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs probe_or/results

module load cuda scipy-stack/2024b
source ~/general/bin/activate
export HF_HOME=/home/meru/links/projects/def-vganesh/meru/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${TARGET}" in
  llama)
    ARGS=(--base_model meta-llama/Meta-Llama-3-8B-Instruct --batch_size 32
          --out probe_or/results/injections_llama.json) ;;
  qwen)
    ARGS=(--base_model Qwen/Qwen3-32B --disable_thinking --batch_size 8
          --out "${OUT:-probe_or/results/injections_qwen.json}") ;;
  *) echo "invalid TARGET=${TARGET}" >&2; exit 2 ;;
esac

# INJ selects which injection set to generate for; OUT where the responses land.
echo "[cfg] TARGET=${TARGET} INJ=${INJ:-default} $(date)"
python -u gen_injections.py "${ARGS[@]}" ${INJ:+--injections "${INJ}"}
echo "[done] ${TARGET} $(date)"
