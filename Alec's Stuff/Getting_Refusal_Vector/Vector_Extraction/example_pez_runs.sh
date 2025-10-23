#!/bin/bash
#
# Example PEZ optimization runs for different use cases
#
# Before running, make sure to:
# 1. Set HUGGING_FACE_HUB_TOKEN in run_pez_optimization.slurm
# 2. Ensure refusal_vector.npz exists at ~/scratch/refusal_vector.npz
#

# ============================================================================
# Example 1: Explore the Refusal Spectrum
# ============================================================================
# Goal: Find prompts at different positions along the refusal direction
# Use case: Understanding how prompts vary from benign to refusal-inducing

echo "Example 1: Sweep across activation values"
for act in -1.0 -0.5 0.0 0.5 1.0; do
  echo "  Submitting jobs for target_activation=${act}"
  sbatch --array=0-4%5 \
    --export=OBJECTIVE=target,TARGET_ACTIVATION=${act},LAYERS=all \
    run_pez_optimization.slurm
done

# ============================================================================
# Example 2: Find Extreme Cases
# ============================================================================
# Goal: Discover prompts that maximally trigger or avoid refusals
# Use case: Adversarial testing, robustness evaluation

echo "Example 2: Find extreme refusal cases"
echo "  Maximizing refusals..."
sbatch --array=0-4%5 \
  --export=OBJECTIVE=maximize,LAYERS=all \
  run_pez_optimization.slurm

echo "  Minimizing refusals..."
sbatch --array=0-4%5 \
  --export=OBJECTIVE=minimize,LAYERS=all \
  run_pez_optimization.slurm

# ============================================================================
# Example 3: Boundary Discovery
# ============================================================================
# Goal: Find prompts near the refusal decision boundary
# Use case: Testing edge cases, understanding classification thresholds

echo "Example 3: Find boundary cases"
sbatch --array=0-9%10 \
  --export=OBJECTIVE=boundary,BOUNDARY_MARGIN=0.1,LAYERS=all \
  run_pez_optimization.slurm

# ============================================================================
# Example 4: Seeded Optimization
# ============================================================================
# Goal: Start from known prompts and optimize them
# Use case: Understanding how small perturbations affect refusal classification

echo "Example 4: Seed-based optimization"

# Optimize a benign prompt toward refusal
sbatch --array=0-2%3 \
  --export=INIT_MODE=seed,SEED_PROMPT=\"What is the weather today?\",OBJECTIVE=maximize \
  run_pez_optimization.slurm

# Optimize a harmful prompt toward benign
sbatch --array=0-2%3 \
  --export=INIT_MODE=seed,SEED_PROMPT=\"How to build explosives\",OBJECTIVE=minimize \
  run_pez_optimization.slurm

# ============================================================================
# Example 5: Mixed Initialization
# ============================================================================
# Goal: Combine seed prompts with random tokens for diverse exploration
# Use case: Finding variations on a theme

echo "Example 5: Mixed initialization"
sbatch --array=0-4%5 \
  --export=INIT_MODE=mixed,SEED_PROMPT=\"Tell me how to\",NUM_RANDOM_TOKENS=15,OBJECTIVE=target,TARGET_ACTIVATION=0.3 \
  run_pez_optimization.slurm

# ============================================================================
# Example 6: Layer Comparison
# ============================================================================
# Goal: Compare how different layer selections affect discovered prompts
# Use case: Understanding which layers are most informative for refusal

echo "Example 6: Compare layer selections"

# All layers averaged
sbatch --array=0-2%3 \
  --export=LAYERS=all,OBJECTIVE=target,TARGET_ACTIVATION=0.5 \
  run_pez_optimization.slurm

# Max L2 layer only
sbatch --array=0-2%3 \
  --export=LAYERS=max,OBJECTIVE=target,TARGET_ACTIVATION=0.5 \
  run_pez_optimization.slurm

# Specific layers
for layer in "8" "16" "24"; do
  sbatch --array=0-2%3 \
    --export=LAYERS=${layer},OBJECTIVE=target,TARGET_ACTIVATION=0.5 \
    run_pez_optimization.slurm
done

# ============================================================================
# Example 7: Hyperparameter Tuning
# ============================================================================
# Goal: Test different optimization settings
# Use case: Finding best configuration for your specific objective

echo "Example 7: Hyperparameter sweep"

# Different learning rates
for lr in 0.05 0.1 0.2; do
  sbatch --array=0-2%3 \
    --export=LEARNING_RATE=${lr},OBJECTIVE=target,TARGET_ACTIVATION=0.0 \
    run_pez_optimization.slurm
done

# Different prompt lengths
for len in 10 20 30; do
  sbatch --array=0-2%3 \
    --export=PROMPT_LENGTH=${len},OBJECTIVE=maximize \
    run_pez_optimization.slurm
done

# ============================================================================
# Example 8: Quick Test Run
# ============================================================================
# Goal: Fast test with fewer steps to validate setup
# Use case: Debugging, initial exploration

echo "Example 8: Quick test run"
sbatch --array=0-1%2 \
  --export=NUM_STEPS=500,OBJECTIVE=target,TARGET_ACTIVATION=0.0,SAVE_EVERY=50 \
  run_pez_optimization.slurm

# ============================================================================
# Example 9: Long, Thorough Optimization
# ============================================================================
# Goal: Deep optimization for publication-quality results
# Use case: Final experiments, discovering high-quality prompts

echo "Example 9: Thorough optimization"
sbatch --array=0-9%10 \
  --export=NUM_STEPS=10000,LEARNING_RATE=0.05,OBJECTIVE=target,TARGET_ACTIVATION=0.5,SAVE_EVERY=500 \
  run_pez_optimization.slurm

# ============================================================================
# After jobs complete, analyze results:
# ============================================================================

echo ""
echo "============================================================================"
echo "After jobs complete, analyze results with:"
echo "  python analyze_pez_results.py --results_dir ~/scratch/pez_results"
echo ""
echo "Monitor job progress:"
echo "  squeue -u \$USER"
echo "  tail -f slurm-pez_opt-*.out"
echo "============================================================================"
