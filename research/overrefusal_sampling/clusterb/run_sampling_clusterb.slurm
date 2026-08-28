#!/bin/bash
#SBATCH -J or_sampling
#SBATCH -A <ACCOUNT>
#SBATCH -p <GPU_PARTITION>
#SBATCH --qos=<QOS>
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#
# Shard across an array job:
#SBATCH --array=0-49

set -euo pipefail
trap 'echo "[error] failed at line ${LINENO}" >&2' ERR

echo "[stage] job start"

# ---- Scratch locations (avoid HOME quota) ----
export SCRATCH_BASE="${HOME}/scratch"

# IMPORTANT: array jobs can race on shared conda caches.
# Use a per-array-job package cache to avoid corruption, and lock env creation/installs.
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-nojob}}"
export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda_pkgs_${ARRAY_JOB_ID}"
export CONDA_ENVS_PATH="${SCRATCH_BASE}/conda_envs"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${CONDA_PKGS_DIRS}" "${CONDA_ENVS_PATH}" "${PIP_CACHE_DIR}" "${TMPDIR}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

# Lock so only one task creates/updates env at a time.
LOCKFILE="${SCRATCH_BASE}/clare_or_sampling_env.lock"
exec 9>"${LOCKFILE}"
flock -x 9

ENV_PREFIX="${CONDA_ENVS_PATH}/clare-or-sampling"
if [ ! -d "${ENV_PREFIX}" ]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi
conda activate "${ENV_PREFIX}"

echo "[stage] python: $(python --version)"

# ---- Working directory ----
# Default assumes you uploaded this folder into your HOME as:
#   $HOME/clare_sampling
# You can override WORKDIR at submit time if you place it elsewhere.
WORKDIR="${WORKDIR:-${HOME}/clare_sampling}"
if [ ! -d "${WORKDIR}" ]; then
  echo "[error] WORKDIR not found: ${WORKDIR}" >&2
  echo "Upload the Sampling folder to ${WORKDIR} or submit with WORKDIR=/path/to/folder" >&2
  exit 2
fi
cd "${WORKDIR}"

# ---- Dependencies (install once per array via lock) ----
INSTALL_DEPS="${INSTALL_DEPS:-1}"
if [ "${INSTALL_DEPS}" = "1" ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements.txt
fi

# Release the lock (other array tasks can proceed).
flock -u 9

# ---- Inputs / outputs ----
mkdir -p "${WORKDIR}/data"

BENIGN_CSV="${BENIGN_CSV:-data/benign_dolly.csv}"

# Default outputs to SCRATCH (safer than HOME). You can override OUTDIR at submit time.
OUTDIR="${OUTDIR:-${SCRATCH_BASE}/clare_sampling_outputs}"
mkdir -p "${OUTDIR}"

# If benign CSV wasn't uploaded, try to build it from a local Dolly JSONL.
if [ ! -f "${WORKDIR}/${BENIGN_CSV}" ] && [ ! -f "${BENIGN_CSV}" ]; then
  DOLLY_JSONL_CANDIDATES=(
    "${WORKDIR}/data/databricks-dolly-15k.jsonl"
    "${WORKDIR}/databricks-dolly-15k.jsonl"
  )
  DOLLY_JSONL=""
  for c in "${DOLLY_JSONL_CANDIDATES[@]}"; do
    if [ -f "${c}" ]; then
      DOLLY_JSONL="${c}"
      break
    fi
  done

  if [ -n "${DOLLY_JSONL}" ]; then
    echo "[stage] benign CSV missing; building from ${DOLLY_JSONL}"
    python -m sampling.run_prepare_benign \
      --dolly_jsonl "${DOLLY_JSONL}" \
      --out_csv "${BENIGN_CSV}" \
      --include_context 0
  else
    echo "[error] Missing benign prompts CSV: ${BENIGN_CSV}" >&2
    echo "Place one of the following in ${WORKDIR} then resubmit:" >&2
    echo "  - ${WORKDIR}/data/benign_dolly.csv" >&2
    echo "  - OR ${WORKDIR}/data/databricks-dolly-15k.jsonl (script will build the CSV automatically)" >&2
    exit 2
  fi
fi

NUM_SHARDS="${NUM_SHARDS:-50}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"
OUT_JSONL="${OUTDIR}/samples_shard_${SHARD_IDX}.jsonl"

echo "[paths] WORKDIR=${WORKDIR}"
echo "[paths] BENIGN_CSV=${BENIGN_CSV}"
echo "[paths] OUT_JSONL=${OUT_JSONL}"
echo "[shard] ${SHARD_IDX}/${NUM_SHARDS}"

# ---- Provider mix to avoid org-level Anthropic rate limits ----
# Default: always run OpenAI+Gemini+DeepSeek, and only run Anthropic on 1/N shards.
# This keeps total Anthropic RPM low without slowing all shards.
ANTHROPIC_EVERY="${ANTHROPIC_EVERY:-10}"  # 1 out of 10 array tasks includes Anthropic
BASE_PROVIDERS="${BASE_PROVIDERS:-openai,gemini,deepseek}"
PROVIDERS_EFFECTIVE="${PROVIDERS:-${BASE_PROVIDERS}}"
if [ -z "${PROVIDERS:-}" ]; then
  if [ "${ANTHROPIC_EVERY}" -gt 0 ] && [ $((SHARD_IDX % ANTHROPIC_EVERY)) -eq 0 ]; then
    PROVIDERS_EFFECTIVE="${BASE_PROVIDERS},anthropic"
  fi
fi
echo "[providers] ${PROVIDERS_EFFECTIVE}"

python -m sampling.run_generate \
  --benign_csv "${BENIGN_CSV}" \
  --out_jsonl "${OUT_JSONL}" \
  --providers "${PROVIDERS_EFFECTIVE}" \
  --prompt_variants "${PROMPT_VARIANTS:-v1,v2,v3}" \
  --num_samples "${NUM_SAMPLES:-3}" \
  --num_samples_overrides "${NUM_SAMPLES_OVERRIDES:-anthropic=1}" \
  --temperature "${TEMP:-0.95}" \
  --max_output_tokens "${MAX_TOKENS:-256}" \
  --max_output_tokens_overrides "${MAX_TOKENS_OVERRIDES:-anthropic=192}" \
  --retries "${RETRIES:-6}" \
  --retry_base_sleep_s "${RETRY_BASE_SLEEP_S:-2.0}" \
  --skip_provider_errors 1 \
  --num_shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}"

echo "[done] wrote ${OUT_JSONL}"

