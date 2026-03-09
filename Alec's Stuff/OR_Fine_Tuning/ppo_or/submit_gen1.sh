#!/bin/bash
set -euo pipefail

PPO_DIR="$HOME/ClaRE/Alec's Stuff/OR_Fine_Tuning/ppo_or"
cd "$PPO_DIR"

CONFIGS=(
  gen1_baseline
  gen1_high_temp
  gen1_creative_prompt
  gen1_high_temp_creative_rollouts
)

echo "=== Generation 1 Sweep ==="

for cfg_name in "${CONFIGS[@]}"; do
  cfg_file="$PPO_DIR/configs/${cfg_name}.json"
  if [ ! -f "$cfg_file" ]; then
    echo "[skip] $cfg_file not found"
    continue
  fi

  export CONFIG_FILE="$cfg_file"
  export SWEEP_NAME="$cfg_name"

  job_out=$(sbatch --export=ALL --job-name="ppo_${cfg_name}" "$PPO_DIR/run_sweep_job.slurm" 2>&1)
  echo "[submitted] $cfg_name => $job_out"
done

echo ""
echo "=== All submitted ==="
squeue -u "$USER" -o '%.10i %.25j %.8T %.10M %.6D %R'
