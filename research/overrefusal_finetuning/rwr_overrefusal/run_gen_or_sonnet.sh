#!/bin/bash

: "${PROJECT:?set PROJECT to the project directory holding hf_cache/ and the venvs}"
# =============================================================================
# run_gen_or_sonnet.sh -- LOGIN-NODE runner for generate_or_sonnet.py
#
# This is NOT a SLURM job. It uses the Anthropic MESSAGE BATCHES API (Claude
# Sonnet 5) and downloads the yahma/alpaca-cleaned dataset -- both need internet,
# which <cluster> COMPUTE nodes do NOT have. Run it on a LOGIN node:
#
#     export ANTHROPIC_API_KEY=sk-ant-...      # your key
#     bash run_gen_or_sonnet.sh
#
# It submits one batch and POLLS until it ends (batch SLA is up to 24h, but 6000
# short requests usually finish in well under an hour). The batch id is persisted
# to <output_dir>/.batch_id_shard000.json, so if this is killed you can simply
# re-run the SAME command and it resumes polling the SAME batch (no duplicate,
# no double charge). To poll in the background: `nohup bash run_gen_or_sonnet.sh &`.
#
# It costs real API money (~$20 at the 6000x6 default, batch pricing) -- see the
# estimate in the accompanying report before launching a full run. Smoke-test
# first with a small --num_prompts (append it as an arg, see below).
#
# Any args you pass are forwarded to generate_or_sonnet.py, e.g.:
#     bash run_gen_or_sonnet.sh --num_prompts 20 --output_dir ./gen_or_sonnet_smoke
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# --- Modules: no cuda (no GPU used); arrow for datasets/pyarrow. ---
# (Piping `module load` into anything subshells it and loses EBROOT* env vars --
#  see CLAUDE.md gotchas -- so keep these as plain top-level commands.)
module load scipy-stack/2024b gcc arrow/23.0.1

# --- Training venv (has anthropic, datasets, etc.). ---
source ~/general/bin/activate

# --- HF cache: project-space cache. Login node HAS internet, so DO NOT set the
#     HF offline flags -- we need to download alpaca-cleaned and reach the API. ---
export HF_HOME=$PROJECT/hf_cache
export HF_HUB_DISABLE_XET=1

# --- Anthropic key must be in the environment (same convention as the other
#     claude scripts in this repo: judge_with_claude.py / generate_claude_heldout.py). ---
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set. Export it before running, e.g.:" >&2
    echo "    export ANTHROPIC_API_KEY=sk-ant-..." >&2
    exit 1
fi

# --- Generate. Defaults: 6000 prompts x 6 rewrites, model=claude-sonnet-5,
#     poll every 60s, output ./gen_or_sonnet. Override any of these via CLI args. ---
python generate_or_sonnet.py \
    --num_prompts 6000 \
    --n_per_prompt 6 \
    --output_dir ./gen_or_sonnet \
    --seed 43 \
    --model claude-sonnet-5 \
    --poll_interval 60 \
    "$@"
