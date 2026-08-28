#!/bin/bash
#
# Setup script to deploy PEZ optimization on <cluster> cluster
#
# This copies the necessary files to your cluster home directory
# and prepares everything for running PEZ jobs
#

set -e

echo "============================================================================"
echo "Setting up PEZ Optimization on <cluster> Cluster"
echo "============================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Destination directory on cluster (use scratch to avoid issues on login node)
DEST_DIR="${HOME}/scratch/pez_optimization"

echo "Source directory: ${SCRIPT_DIR}"
echo "Destination directory: ${DEST_DIR}"
echo ""

# Create destination directory
echo "[1/3] Creating destination directory..."
mkdir -p "${DEST_DIR}"

# Copy Python script
echo "[2/3] Copying PEZ Python script..."
if [[ -f "${SCRIPT_DIR}/pez_refusal_optimization.py" ]]; then
  cp "${SCRIPT_DIR}/pez_refusal_optimization.py" "${DEST_DIR}/"
  echo "  ✓ Copied pez_refusal_optimization.py"
else
  echo "  ✗ ERROR: pez_refusal_optimization.py not found in ${SCRIPT_DIR}"
  exit 1
fi

# Copy SLURM script
echo "[3/3] Copying SLURM submission script..."
if [[ -f "${SCRIPT_DIR}/run_pez_optimization.slurm" ]]; then
  cp "${SCRIPT_DIR}/run_pez_optimization.slurm" "${DEST_DIR}/"
  echo "  ✓ Copied run_pez_optimization.slurm"
else
  echo "  ✗ ERROR: run_pez_optimization.slurm not found in ${SCRIPT_DIR}"
  exit 1
fi

# Make scripts executable
chmod +x "${DEST_DIR}/pez_refusal_optimization.py"
chmod +x "${DEST_DIR}/run_pez_optimization.slurm"

echo ""
echo "============================================================================"
echo "Setup Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Navigate to the PEZ directory:"
echo "   cd ${DEST_DIR}"
echo ""
echo "2. Edit the SLURM script to add your HuggingFace token (if not already set):"
echo "   nano run_pez_optimization.slurm"
echo "   (Set HUGGING_FACE_HUB_TOKEN on line 29)"
echo ""
echo "3. Submit a test job:"
echo "   sbatch --array=0-1%2 --export=NUM_STEPS=100 run_pez_optimization.slurm"
echo ""
echo "4. Monitor progress:"
echo "   tail -f slurm-pez_opt-*.out"
echo ""
echo "5. Check results:"
echo "   ls ${HOME}/scratch/pez_results/"
echo ""
echo "============================================================================"
