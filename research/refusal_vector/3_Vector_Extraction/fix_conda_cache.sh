#!/bin/bash
#
# Fix corrupted conda cache on PACE cluster
#
# Run this if you see errors like:
#   InvalidArchiveError
#   Stale file handle
#   seeking backwards is not allowed
#

echo "=================================================="
echo "Fixing Corrupted Conda Cache"
echo "=================================================="

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "Step 1: Cleaning conda cache..."
conda clean -y --all 2>/dev/null || true

echo ""
echo "Step 2: Removing corrupted package files..."
rm -rf "${HOME}/.conda/pkgs/ncurses-"* 2>/dev/null || true
rm -rf "${HOME}/.conda/pkgs/openssl-"* 2>/dev/null || true
rm -rf "${HOME}/.conda/pkgs/python-"* 2>/dev/null || true
rm -rf "${HOME}/.conda/pkgs/pip-"* 2>/dev/null || true
rm -rf "${HOME}/.conda/pkgs/setuptools-"* 2>/dev/null || true
rm -rf "${HOME}/.conda/pkgs/tk-"* 2>/dev/null || true

echo ""
echo "Step 3: Cleaning conda cache again..."
conda clean -y --all 2>/dev/null || true

echo ""
echo "Step 4: Removing existing llama8b-env (if exists)..."
conda env remove -n llama8b-env -y 2>/dev/null || true

echo ""
echo "Step 5: Creating fresh environment..."
conda create -y -n llama8b-env python=3.10

echo ""
echo "=================================================="
echo "Done! Conda cache has been cleaned."
echo "=================================================="
echo ""
echo "The SLURM script should now work properly."
echo "You can test by submitting a job:"
echo "  sbatch --array=0-1%2 --export=NUM_STEPS=100 run_pez_optimization.slurm"
