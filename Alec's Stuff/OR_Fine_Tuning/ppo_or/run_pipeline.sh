#!/bin/bash
# Full OR pipeline: paraphrase array → merge & select → PPO training
# Usage: bash run_pipeline.sh  (from the ppo_or/ directory on PACE)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[pipeline] Script dir: ${SCRIPT_DIR}"

# ---- Step 1: Submit paraphrase array job ----
echo "[pipeline] Submitting paraphrase array job..."
PARA_JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/run_paraphrase_array.slurm")
echo "[pipeline] Paraphrase array job: ${PARA_JOB_ID}"

# The output dir uses the array job ID
PARA_OUTDIR="${HOME}/scratch/ppo_or_data/or_paraphrase_array_${PARA_JOB_ID}"

# ---- Step 2: Submit merge+select job (depends on all paraphrase shards) ----
echo "[pipeline] Submitting merge+select job (depends on ${PARA_JOB_ID})..."
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:${PARA_JOB_ID} <<MERGEEOF
#!/bin/bash
#SBATCH -J or_merge_select
#SBATCH -A coc
#SBATCH -p ice-cpu
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --output=slurm-merge-%j.out
#SBATCH --error=slurm-merge-%j.err

set -euo pipefail

module load anaconda3
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "\${HOME}/scratch/conda_envs/ppo-or-env"

SHARD_DIR="${PARA_OUTDIR}"
JSONL_OUT="${PARA_OUTDIR}/top20pct_or_prompts.jsonl"

echo "[merge] Shard dir: \${SHARD_DIR}"
echo "[merge] Output JSONL: \${JSONL_OUT}"

python -u "${SCRIPT_DIR}/merge_and_select.py" \\
    --shard_dir "\${SHARD_DIR}" \\
    --num_shards 60 \\
    --min_paraphrases 20 \\
    --output_jsonl "\${JSONL_OUT}"

# Also copy to the standard data location for the training job
cp "\${JSONL_OUT}" "\${HOME}/scratch/ppo_or_data/top20pct_or_prompts.jsonl"
echo "[merge] Copied to \${HOME}/scratch/ppo_or_data/top20pct_or_prompts.jsonl"
echo "[merge] Done"
MERGEEOF
)
echo "[pipeline] Merge+select job: ${MERGE_JOB_ID}"

# ---- Step 3: Submit PPO training job (depends on merge) ----
echo "[pipeline] Submitting PPO training job (depends on ${MERGE_JOB_ID})..."
PPO_JOB_ID=$(sbatch --parsable --dependency=afterok:${MERGE_JOB_ID} \
    "${SCRIPT_DIR}/run_ppo_or_training_pace.slurm")
echo "[pipeline] PPO training job: ${PPO_JOB_ID}"

echo ""
echo "========================================"
echo "Pipeline submitted!"
echo "  1. Paraphrase array: ${PARA_JOB_ID} (60 shards)"
echo "  2. Merge+select:     ${MERGE_JOB_ID} (after paraphrase)"
echo "  3. PPO training:     ${PPO_JOB_ID} (after merge)"
echo ""
echo "Monitor with: squeue -u \$(whoami)"
echo "Paraphrase output: ${PARA_OUTDIR}"
echo "========================================"
