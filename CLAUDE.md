# ClaRE - Classification-guided Latent Representation Elicitation

## Project Purpose

ClaRE is a white-box automated red-teaming research project focused on **over-refusal** in LLMs. The goal is to use mechanistic interpretability (specifically refusal direction vectors extracted from model activations) to train an attack model that generates prompts which are semantically similar to benign seeds but reliably trigger over-refusal in target models.

### Core Idea

1. Extract a **refusal direction vector** from a target model's activation space (difference-of-means between refusal and benign activations).
2. Generate candidate prompt rewrites that maximize alignment with this refusal vector while preserving semantic meaning.
3. Use a **reward signal** combining:
   - **Refusal activation score**: cosine/dot-product similarity between prompt activations and the refusal vector (higher = more likely to trigger refusal)
   - **Semantic similarity penalty**: subtracted to keep generated prompts close to the original seed meaning
   - **Harmfulness filter**: reject anything actually harmful (Llama Guard)
4. Train via PPO or select via SFT scoring to produce a dataset/model that can systematically find safety failures.

## Repository Layout

```
research/
  refusal_vector/           # Refusal direction extraction pipeline
    Vector_Extraction/      # Core: extract_activations_sharded.py, compute_refusal_vector.py
    Steering_Demo/          # Demo of steering with refusal vector
  refusal_classification/   # Classify model outputs as refusals (M3-optimized, regex)
  prompt_generations/       # LLaMA prompt generation and filtering
  epo_dreams/               # Evolutionary Prompt Optimization for refusal
  overrefusal_finetuning/   # PPO training for over-refusal prompt rewriting
    ppo_or/                 # Main PPO pipeline (train_ppo.py, reward_model.py, config.py)
    ppo_refusal/            # Refusal-maximization variant
  overrefusal_sampling/     # Multi-provider (OpenAI/Anthropic/Gemini/DeepSeek) candidate generation
SFT_Scoring/                # Scoring pipeline: semantic_similarity + refusal_activation + harmfulness
src/                        # Shared config (dataset sources) and utils (text cleaning, dedup)
scripts/                    # CSV processing helpers
```

## Key Files

- `research/refusal_vector/Vector_Extraction/compute_refusal_vector.py` - Computes refusal direction as difference-in-means
- `research/refusal_vector/Vector_Extraction/refusal_vector.layer032.npz` - Pre-computed refusal vector (layer 32), used by SFT_Scoring
- `research/overrefusal_finetuning/ppo_or/train_ppo.py` - Main PPO training loop (~644 lines)
- `research/overrefusal_finetuning/ppo_or/reward_model.py` - Reward: `e^(9.2*(similarity-0.5)) * (rewrite_refusal - original_refusal) / 10`
- `research/overrefusal_finetuning/ppo_or/config.py` - Dataclass config (Llama-3-8B-Instruct, LoRA r=32, few-shot examples)
- `SFT_Scoring/pipeline.py` - Orchestrates scoring and filtering
- `SFT_Scoring/config.yaml` - Thresholds: harmfulness >= 0.3 reject, similarity <= 0.5 reject, top 5% by refusal activation

## Sister Repository

`../ClaRE_ALEC/` is Alec's working branch. It has:
- `Alec's Stuff/` directory with OR_Fine_Tuning, Getting_Refusal_Vector, Sampling, EPO_Dreams
- `Andrew Refusal Classification/` directory
- Extra files not yet in main: `merge_and_select.py`, `paraphrase_and_score.py`, `or_paraphrase_3k/` dataset
- Slightly different reward formula coefficient (18.4 vs 9.2 in exponential)

## Tech Stack

- **Models**: Llama 3 8B Instruct (base), Llama Guard 3 8B (harmfulness), DistilRoBERTa (classification)
- **Training**: PyTorch, HuggingFace Transformers/TRL, PEFT/LoRA, PPO
- **Compute**: SLURM/PACE cluster (H200 GPUs), Apple M3 local optimization (MLX)
- **APIs**: OpenAI, Anthropic, Google Gemini, DeepSeek (for candidate sampling)

## Branch Strategy

- `sample-data` is the active main branch (the `main` branch is ~60 commits behind)
- Large CSV/data files are tracked in the repo (total ~455MB)

## RWR Offline RL Pipeline (research/overrefusal_finetuning/rwr_overrefusal/)

Reward-Weighted Regression with quantile-binned sampling, trained on the `or_paraphrase_3k` dataset (34,853 pairs across 3,000 seed prompts). Uses `or_score_raw` (unclamped) as the reward signal — this is `exp(18.4*(sim-0.75)) * refusal_delta / 100`, which preserves the mech interp signal (dot product with refusal vector) while gating on semantic similarity.

### Steps

1. **Data pipeline** (`rwr_data.py`) — Load shards, filter (sim < 0.5), quantile-bin rewards, weighted sampler
2. **Training** (`train_rwr.py`) — Llama-3-8B-Instruct + LoRA, causal LM fine-tuning with bin-weighted sampling
3. **Config** (`rwr_config.py`) — Model, LoRA, binning, and training hyperparameters
4. **Evaluation** (`eval_rwr.py`) — Generate from held-out prompts, score with ORRewardModel, compare distributions

## Development Notes

- Per-module requirements.txt files (no single top-level one)
- SLURM job templates co-located with their respective modules
- Activation extraction is sharded for distributed compute
- M3-optimized classifier uses chunked processing with 20-min resumable checkpoints
