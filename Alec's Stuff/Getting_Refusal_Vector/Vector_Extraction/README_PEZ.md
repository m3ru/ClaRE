# PEZ Optimization for Refusal Vector Discovery

This directory contains an implementation of **PEZ (Prompt Embedding Zeroization)** adapted for discovering prompts with specific refusal activation characteristics.

## What is PEZ?

PEZ is a gradient-based discrete prompt optimization method from ["Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery"](https://arxiv.org/abs/2302.03668) (Wen et al., 2023).

**Key Idea**: Unlike soft prompt tuning which optimizes continuous embeddings, or discrete methods which struggle with non-differentiable token selection, PEZ:

1. Maintains **continuous embeddings** during optimization
2. **Projects to discrete tokens** during the forward pass
3. Computes gradients and **updates the continuous embeddings**
4. Outputs **interpretable text prompts** at the end

This allows gradient-based optimization while producing human-readable prompts.

## Our Application: Refusal Vector Optimization

We use PEZ to discover prompts that achieve specific activation levels along the **refusal direction** in Llama-3-8B-Instruct's representation space.

### Use Cases

1. **Maximize refusals**: Find prompts that strongly trigger refusal behavior
2. **Minimize refusals**: Discover prompts that avoid triggering refusals
3. **Target specific levels**: Find prompts at precise positions along the refusal direction
4. **Boundary cases**: Discover prompts near the refusal decision boundary

### How It Works

```
Input: Refusal vector (mean_refusal - mean_benign activations)
       Frozen Llama-3-8B-Instruct model
       Optimization objective (e.g., target activation = 0.5)

Process: For each optimization step:
  1. Project continuous embeddings → discrete tokens
  2. Run model forward pass → extract activations
  3. Compute dot product with refusal vector
  4. Compute loss based on objective
  5. Backprop gradients to continuous embeddings
  6. Update embeddings with optimizer (AdamW)

Output: Discrete text prompt with desired refusal characteristics
```

## Files

- **`pez_refusal_optimization.py`**: Main PEZ implementation
- **`run_pez_optimization.slurm`**: SLURM script for compute cluster
- **`README_PEZ.md`**: This file

## Installation

The SLURM script automatically installs dependencies. For local testing:

```bash
pip install torch transformers accelerate sentencepiece numpy matplotlib
```

## Usage

### Basic Example (Local)

```bash
python pez_refusal_optimization.py \
  --refusal_vector ~/scratch/refusal_vector.npz \
  --objective target \
  --target_activation 0.5 \
  --num_steps 1000 \
  --prompt_length 20 \
  --output_dir ./pez_results
```

### SLURM Usage

#### 1. Edit Configuration

First, add your HuggingFace token to `run_pez_optimization.slurm`:

```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

#### 2. Basic Submission

Run with default settings (target activation = 0.0):

```bash
sbatch --array=0-9%10 run_pez_optimization.slurm
```

This runs 10 parallel jobs with different random seeds.

#### 3. Different Objectives

**Maximize refusals:**
```bash
sbatch --array=0-4%5 --export=OBJECTIVE=maximize run_pez_optimization.slurm
```

**Minimize refusals:**
```bash
sbatch --array=0-4%5 --export=OBJECTIVE=minimize run_pez_optimization.slurm
```

**Target specific activation:**
```bash
sbatch --array=0-9%10 --export=OBJECTIVE=target,TARGET_ACTIVATION=0.75 run_pez_optimization.slurm
```

**Boundary cases:**
```bash
sbatch --array=0-9%10 --export=OBJECTIVE=boundary,BOUNDARY_MARGIN=0.1 run_pez_optimization.slurm
```

#### 4. Different Initializations

**Random initialization (default):**
```bash
sbatch --array=0-4%5 --export=INIT_MODE=random run_pez_optimization.slurm
```

**Seed from existing prompt:**
```bash
sbatch --array=0-4%5 \
  --export=INIT_MODE=seed,SEED_PROMPT="How do I build a bomb?" \
  run_pez_optimization.slurm
```

**Mixed (seed + random):**
```bash
sbatch --array=0-4%5 \
  --export=INIT_MODE=mixed,SEED_PROMPT="Tell me how to",NUM_RANDOM_TOKENS=10 \
  run_pez_optimization.slurm
```

#### 5. Layer Selection

**Use all layers (averaged):**
```bash
sbatch --array=0-4%5 --export=LAYERS=all run_pez_optimization.slurm
```

**Use max L2 layer only:**
```bash
sbatch --array=0-4%5 --export=LAYERS=max run_pez_optimization.slurm
```

**Use specific layers:**
```bash
sbatch --array=0-4%5 --export=LAYERS="8,16,24" run_pez_optimization.slurm
```

#### 6. Advanced Configuration

Combine multiple settings:

```bash
sbatch --array=0-19%10 \
  --export=OBJECTIVE=target,TARGET_ACTIVATION=0.3,LAYERS=all,NUM_STEPS=5000,LEARNING_RATE=0.05 \
  run_pez_optimization.slurm
```

## Configuration Options

### Model Settings
- `--model`: HuggingFace model name (default: `meta-llama/Meta-Llama-3-8B-Instruct`)
- `--dtype`: Model precision (`bf16`, `fp16`, `fp32`)

### Refusal Vector Settings
- `--refusal_vector`: Path to `refusal_vector.npz` (required)
- `--layers`: Which layers to use (`all`, `max`, or comma-separated indices)

### Optimization Settings
- `--num_steps`: Number of optimization steps (default: 3000)
- `--learning_rate`: Learning rate (default: 0.1)
- `--prompt_length`: Number of tokens to optimize (default: 20)

### Objective Settings
- `--objective`: Optimization goal
  - `maximize`: Find prompts that maximize refusal
  - `minimize`: Find prompts that minimize refusal
  - `target`: Target specific activation value
  - `boundary`: Find prompts near decision boundary
- `--target_activation`: Target value for `target` objective (default: 0.0)
- `--boundary_margin`: Margin for `boundary` objective (default: 0.1)

### Initialization Settings
- `--init_mode`: How to initialize embeddings
  - `random`: Random tokens from vocabulary
  - `seed`: Start from provided prompt
  - `mixed`: Combine seed prompt with random tokens
- `--seed_prompt`: Seed prompt for `seed` or `mixed` modes
- `--num_random_tokens`: Number of random tokens for `mixed` mode

### Optimizer Settings
- `--optimizer`: Optimizer type (`adamw`, `sgd`)
- `--weight_decay`: Weight decay (default: 0.01)
- `--gradient_clip`: Gradient clipping value (default: 1.0)

### Output Settings
- `--save_every`: Save checkpoint every N steps (default: 100)
- `--output_dir`: Output directory
- `--run_name`: Name for this run (auto-generated if not provided)

### Compute Settings
- `--device`: Device to use (`cuda`, `cpu`)
- `--seed`: Random seed for reproducibility

## Output

Each run creates a directory with:

```
pez_results/
  pez_target_random_20250123_143022/
    ├── config.json                    # Full configuration
    ├── results.json                   # Optimization history
    ├── final_embeddings.pt            # Final continuous embeddings
    ├── optimization_curves.png        # Loss and activation curves
    └── checkpoint_step_*.pt           # Periodic checkpoints
```

### Interpreting Results

**`results.json`** contains:
- `best_prompt`: The discovered prompt with best loss
- `best_activation`: Its refusal activation score
- `best_step`: When it was found
- `losses`: Loss at each step
- `activations`: Refusal activation at each step
- `prompts`: Prompts at checkpoints

**Activation values:**
- Positive: Aligned with refusal direction
- Negative: Aligned with benign direction
- Near zero: Neutral or boundary cases

## Example Workflows

### 1. Explore the Refusal Spectrum

Run multiple jobs targeting different activation levels:

```bash
# Create a sweep over activation values
for act in -1.0 -0.5 0.0 0.5 1.0; do
  sbatch --array=0-4%5 \
    --export=OBJECTIVE=target,TARGET_ACTIVATION=${act} \
    run_pez_optimization.slurm
done
```

### 2. Find Diverse Boundary Cases

Use boundary objective with different random seeds:

```bash
sbatch --array=0-19%10 \
  --export=OBJECTIVE=boundary,BOUNDARY_MARGIN=0.05 \
  run_pez_optimization.slurm
```

### 3. Test Robustness of Existing Prompts

Start from known prompts and optimize:

```bash
# Test a benign prompt
sbatch --array=0-4%5 \
  --export=INIT_MODE=seed,SEED_PROMPT="What is the weather today?",OBJECTIVE=maximize \
  run_pez_optimization.slurm

# Test a potentially harmful prompt
sbatch --array=0-4%5 \
  --export=INIT_MODE=seed,SEED_PROMPT="How to make explosives",OBJECTIVE=minimize \
  run_pez_optimization.slurm
```

### 4. Layer-specific Analysis

Compare results from different layers:

```bash
# Max L2 layer
sbatch --array=0-4%5 --export=LAYERS=max,OBJECTIVE=target,TARGET_ACTIVATION=0.5 run_pez_optimization.slurm

# All layers averaged
sbatch --array=0-4%5 --export=LAYERS=all,OBJECTIVE=target,TARGET_ACTIVATION=0.5 run_pez_optimization.slurm

# Specific layers
for layer in "8" "16" "24" "32"; do
  sbatch --array=0-4%5 --export=LAYERS=${layer},OBJECTIVE=target,TARGET_ACTIVATION=0.5 run_pez_optimization.slurm
done
```

## Monitoring Progress

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f slurm-pez_opt-JOBID-ARRAYID.out
```

Check results:
```bash
ls ~/scratch/pez_results/
cat ~/scratch/pez_results/pez_*/results.json | grep best_prompt
```

## Tips and Best Practices

1. **Start with fewer steps** (e.g., 1000) to test configurations quickly
2. **Use array jobs** to explore multiple seeds in parallel
3. **Monitor the first few runs** to ensure reasonable convergence
4. **Adjust learning rate** if loss plateaus or oscillates wildly
5. **Try different prompt lengths** - longer prompts may be more expressive
6. **Layer averaging** (`LAYERS=all`) tends to be more stable than single layers
7. **Save checkpoints frequently** during long runs for resumability

## Troubleshooting

**Out of memory:**
- Reduce `--prompt_length`
- Use smaller batch size (currently fixed at 1)
- Change `--dtype` to `fp16`

**Poor convergence:**
- Increase `--num_steps`
- Adjust `--learning_rate` (try 0.05 or 0.2)
- Try different `--init_mode`
- Check if refusal vector loaded correctly

**Job fails immediately:**
- Verify `HUGGING_FACE_HUB_TOKEN` is set
- Check refusal vector path exists
- Ensure model name is correct

## Citation

If you use this implementation, please cite the original PEZ paper:

```bibtex
@article{wen2023hard,
  title={Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery},
  author={Wen, Yuxin and Jain, Neel and Kirchenbauer, John and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom},
  journal={arXiv preprint arXiv:2302.03668},
  year={2023}
}
```

## Advanced: Multi-Objective Optimization

For discovering diverse prompts across multiple objectives, you can submit a meta-job:

```bash
#!/bin/bash
# submit_pez_sweep.sh

OBJECTIVES=("maximize" "minimize" "boundary")
ACTIVATIONS=("-0.5" "0.0" "0.5")
LAYERS=("all" "max")

for obj in "${OBJECTIVES[@]}"; do
  for layer in "${LAYERS[@]}"; do
    if [[ "$obj" == "target" ]]; then
      for act in "${ACTIVATIONS[@]}"; do
        sbatch --array=0-4%5 \
          --export=OBJECTIVE=$obj,TARGET_ACTIVATION=$act,LAYERS=$layer \
          run_pez_optimization.slurm
      done
    else
      sbatch --array=0-4%5 \
        --export=OBJECTIVE=$obj,LAYERS=$layer \
        run_pez_optimization.slurm
    fi
  done
done
```

Then run:
```bash
chmod +x submit_pez_sweep.sh
./submit_pez_sweep.sh
```

This creates a comprehensive dataset of optimized prompts across the refusal spectrum.
