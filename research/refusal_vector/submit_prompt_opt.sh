#!/bin/bash
# Fire the whole prompt-optimization experiment on PACE in one shot.
#
# PACE was in quarterly maintenance 2026-08-11 06:00 -> 2026-08-13 23:59, so this could not
# be submitted when the code was written. Run it from the laptop once PACE is back:
#     bash research/refusal_vector/submit_prompt_opt.sh
#
# Chain: pull repo on PACE -> build all-layer directions -> array of 6 optimization configs
# (the array waits on the build via --dependency=afterok, so both can be queued immediately).
set -euo pipefail
REMOTE="${1:-pace}"

echo "[1/3] checking PACE + syncing scripts"
# NOTE: ~/ClaRE on PACE is NOT a git checkout (files are copied up), so sync by scp.
ssh -o ConnectTimeout=20 "${REMOTE}" 'mkdir -p ~/ClaRE/research/refusal_vector'
scp -q "$(dirname "$0")"/{build_harmful_dirs.py,prompt_opt.py,run_build_dirs_pace.slurm,run_prompt_opt_pace.slurm} \
    "${REMOTE}:~/ClaRE/research/refusal_vector/"

echo "[2/3] submitting direction build"
BUILD_ID=$(ssh "${REMOTE}" 'cd ~/ClaRE/research/refusal_vector && sbatch --parsable run_build_dirs_pace.slurm')
echo "      build job: ${BUILD_ID}"

echo "[3/3] submitting optimization array (waits on build)"
ARRAY_ID=$(ssh "${REMOTE}" "cd ~/ClaRE/research/refusal_vector && sbatch --parsable --dependency=afterok:${BUILD_ID} run_prompt_opt_pace.slurm")
echo "      array job: ${ARRAY_ID}  (6 configs: {direction@L17, direction@L12, icannot} x {gcg, pez})"

echo
echo "watch:   ssh ${REMOTE} 'squeue -u aharris345'"
echo "results: ssh ${REMOTE} 'ls -la ~/scratch/prompt_opt/'"
