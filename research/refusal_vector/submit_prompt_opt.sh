#!/bin/bash
# Fire the whole prompt-optimization experiment on <cluster> in one shot.
#
# Run it from a local machine, passing the ssh host to submit on as $1:
#     bash research/refusal_vector/submit_prompt_opt.sh <ssh-host>
#
# Chain: pull repo on <cluster> -> build all-layer directions -> array of 6 optimization configs
# (the array waits on the build via --dependency=afterok, so both can be queued immediately).
set -euo pipefail
REMOTE="${1:?pass the ssh host to submit on}"

echo "[1/3] checking <cluster> + syncing scripts"
# NOTE: ~/anon-repo on <cluster> is NOT a git checkout (files are copied up), so sync by scp.
ssh -o ConnectTimeout=20 "${REMOTE}" 'mkdir -p ~/anon-repo/research/refusal_vector'
scp -q "$(dirname "$0")"/{build_harmful_dirs.py,prompt_opt.py,run_build_dirs_clusterb.slurm,run_prompt_opt_clusterb.slurm} \
    "${REMOTE}:~/anon-repo/research/refusal_vector/"

echo "[2/3] submitting direction build"
BUILD_ID=$(ssh "${REMOTE}" 'cd ~/anon-repo/research/refusal_vector && sbatch --parsable run_build_dirs_clusterb.slurm')
echo "      build job: ${BUILD_ID}"

echo "[3/3] submitting optimization array (waits on build)"
ARRAY_ID=$(ssh "${REMOTE}" "cd ~/anon-repo/research/refusal_vector && sbatch --parsable --dependency=afterok:${BUILD_ID} run_prompt_opt_clusterb.slurm")
echo "      array job: ${ARRAY_ID}  (6 configs: {direction@L17, direction@L12, icannot} x {gcg, pez})"

echo
echo "watch:   ssh ${REMOTE} 'squeue -u <collaborator>'"
echo "results: ssh ${REMOTE} 'ls -la ~/scratch/prompt_opt/'"
