#!/bin/bash
# =============================================================================
# SFT Scoring Pipeline Runner
# =============================================================================
# This script runs the candidate scoring pipeline for SFT training data selection.
#
# Usage:
#   ./run_pipeline.sh                    # Run with defaults
#   ./run_pipeline.sh --config my.yaml   # Run with custom config
#   ./run_pipeline.sh --fresh            # Ignore checkpoints, start fresh
#
# For SLURM: Uncomment the SBATCH directives below and submit with:
#   sbatch run_pipeline.sh
# =============================================================================

# =============================================================================
# SLURM Configuration (uncomment for cluster submission)
# =============================================================================
# #SBATCH --job-name=sft-scoring
# #SBATCH --output=logs/sft_scoring_%j.out
# #SBATCH --error=logs/sft_scoring_%j.err
# #SBATCH --time=24:00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --gres=gpu:h200:1
# #SBATCH --partition=gpu
#
# # Alternative GPU specifications:
# # #SBATCH --gres=gpu:a100:1
# # #SBATCH --gres=gpu:1 --constraint=h200

# =============================================================================
# Environment Setup
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log files (used when running locally, SLURM uses its own)
STDOUT_LOG="logs/pipeline_${TIMESTAMP}.out"
STDERR_LOG="logs/pipeline_${TIMESTAMP}.err"

# =============================================================================
# Environment Activation
# =============================================================================

# Option 1: Conda environment (uncomment and modify as needed)
# if command -v conda &> /dev/null; then
#     echo "Activating conda environment..."
#     source "$(conda info --base)/etc/profile.d/conda.sh"
#     conda activate clare  # Change to your environment name
# fi

# Option 2: Virtual environment (uncomment and modify as needed)
# if [ -d "venv" ]; then
#     echo "Activating virtual environment..."
#     source venv/bin/activate
# elif [ -d "../venv" ]; then
#     source ../venv/bin/activate
# fi

# Option 3: Module system (for HPC clusters, uncomment as needed)
# module load python/3.10
# module load cuda/12.1

# =============================================================================
# HuggingFace Token
# =============================================================================

# Set HF token if not already set (required for gated models like Llama)
# Option 1: Set directly (not recommended for shared scripts)
# export HF_TOKEN="your_token_here"

# Option 2: Load from file
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.huggingface/token")
    echo "Loaded HF token from ~/.huggingface/token"
fi

# Option 3: Prompt if not set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Gated models may fail to load."
    echo "Set with: export HF_TOKEN=your_token"
fi

# =============================================================================
# Configuration
# =============================================================================

# Default paths (modify as needed)
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
CANDIDATES_PATH=""  # Leave empty to use config file value
OUTPUT_DIR=""       # Leave empty to use config file value

# Default thresholds (leave empty to use config file values)
HARMFULNESS_THRESHOLD=""
SIMILARITY_THRESHOLD=""
TOP_K_PERCENT=""

# Batch sizes (adjust based on GPU memory)
BATCH_SIZE=""       # For Llama 8B (semantic similarity + refusal activation)
BATCH_SIZE_GUARD="" # For Llama Guard (harmfulness)

# =============================================================================
# Build Command
# =============================================================================

CMD="python ${SCRIPT_DIR}/pipeline.py"

# Add config file if specified
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

# Add optional overrides
[ -n "$CANDIDATES_PATH" ] && CMD="$CMD --candidates $CANDIDATES_PATH"
[ -n "$OUTPUT_DIR" ] && CMD="$CMD --output $OUTPUT_DIR"
[ -n "$HARMFULNESS_THRESHOLD" ] && CMD="$CMD --harmfulness-threshold $HARMFULNESS_THRESHOLD"
[ -n "$SIMILARITY_THRESHOLD" ] && CMD="$CMD --similarity-threshold $SIMILARITY_THRESHOLD"
[ -n "$TOP_K_PERCENT" ] && CMD="$CMD --top-k-percent $TOP_K_PERCENT"
[ -n "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE"

# Pass through any command line arguments
CMD="$CMD $@"

# =============================================================================
# GPU Configuration
# =============================================================================

# Set CUDA device (useful when multiple GPUs available)
# export CUDA_VISIBLE_DEVICES=0

# PyTorch memory settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# =============================================================================
# Run Pipeline
# =============================================================================

echo "============================================================"
echo "SFT Scoring Pipeline"
echo "============================================================"
echo "Start time: $(date)"
echo "Script directory: $SCRIPT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Log files: $STDOUT_LOG, $STDERR_LOG"
echo ""
echo "Command: $CMD"
echo "============================================================"
echo ""

# Check if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"
    echo "Node: $SLURM_NODELIST"
    echo "GPUs: $SLURM_GPUS_ON_NODE"
    echo ""

    # Print GPU info
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv
        echo ""
    fi

    # Run directly (SLURM handles output redirection)
    $CMD
    EXIT_CODE=$?
else
    # Running locally - redirect output to log files
    echo "Running locally (logging to $STDOUT_LOG)"
    echo ""

    # Print GPU info if available
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv
        echo ""
    fi

    # Run with output redirection
    $CMD 2>&1 | tee "$STDOUT_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
fi

# =============================================================================
# Completion
# =============================================================================

echo ""
echo "============================================================"
echo "Pipeline finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
