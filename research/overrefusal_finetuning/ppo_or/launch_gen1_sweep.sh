#!/bin/bash
# Launch Generation 1 hyperparameter sweep — submit from the ppo_or/ directory
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
SLURM_SCRIPT="${SCRIPT_DIR}/run_sweep_job.slurm"

CONFIGS=(
  "gen1_baseline"
  "gen1_high_temp"
  "gen1_creative_prompt"
  "gen1_high_temp_creative_rollouts"
)

echo "=== Generation 1 Hyperparameter Sweep ==="
echo "Submitting ${#CONFIGS[@]} jobs..."
echo ""

for cfg_name in "${CONFIGS[@]}"; do
  cfg_file="${CONFIGS_DIR}/${cfg_name}.json"
  if [ ! -f "${cfg_file}" ]; then
    echo "[skip] Config not found: ${cfg_file}"
    continue
  fi
  job_id=$(CONFIG_FILE="${cfg_file}" SWEEP_NAME="${cfg_name}" \
    sbatch --export=ALL,CONFIG_FILE="${cfg_file}",SWEEP_NAME="${cfg_name}" \
           --job-name="ppo_${cfg_name}" \
           "${SLURM_SCRIPT}" 2>&1 | grep -o '[0-9]*')
  echo "[submitted] ${cfg_name} => job ${job_id}"
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs: slurm-<jobid>.out / .err"
