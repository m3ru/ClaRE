# Refusal Vector Extraction

Extract a linear "refusal direction" from Llama-3-8B-Instruct hidden states using difference-in-means. This vector is the core feature used by the OR scoring pipeline and RWR training.

## Pipeline Overview

Three sequential stages take raw seed prompts → labeled refusal/benign sets → refusal vector:

```
Reddit prompts
      │
      ▼
┌──────────────────────────┐
│ 1_Prompt_Generation      │   Run Llama-3-8B-Instruct on seed prompts
│                          │   → prompt,response pairs (~80k)
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 2_Refusal_Classification │   ML (ProtectAI) + regex → final_refusals.csv
│                          │   and final_benign.csv
└──────────────────────────┘
      │
      ▼
┌──────────────────────────┐
│ 3_Vector_Extraction      │   Collect per-layer activations → difference
│                          │   of means → refusal_vector.npz
└──────────────────────────┘
```

## Method

1. Run the model on two prompt sets: one that reliably triggers refusals, one that produces normal responses.
2. Collect hidden-state activations at each transformer layer.
3. Compute the refusal vector as:
   ```
   refusal_vector = mean(refusal_activations) - mean(benign_activations)
   ```
4. Select the layer with the largest L2 norm (layer 32 for Llama-3-8B-Instruct).

The resulting vector can score any new prompt by projecting its activations onto this direction.

## Stage 1: Prompt Generation (`1_Prompt_Generation/`)

Runs Llama-3-8B-Instruct on a large pool of seed prompts to produce the prompt/response pairs that will be labeled in stage 2.

| File | Purpose |
|------|---------|
| `Final-People-Reddit-Prompts-SINGLE-COLUMN.csv` | Seed prompts (one per line, Reddit-sourced) |
| `Final-People-Reddit-Prompts-Multiple-Wrappers(in).csv` | Same prompts with multiple instruction wrappers |
| `run_llama_prompts_lines_sharded.py` | Shard-aware runner: generates responses for one prompt-per-line input |
| `run_llama_prompts_reddit.slurm` | SLURM array job for the Reddit set |
| `run_llama_prompts_filtered.slurm` | SLURM array job for the filtered set |
| `run_refusal_filter.py` | Post-hoc filter over generated responses |
| `run_refusal_filter.slurm` | SLURM wrapper for the filter |
| `outputs_reddit.csv` | Raw Llama-8B outputs (small) |
| `llama8b_outputs_filtered.csv` | Merged + filtered Llama-8B outputs (~80k prompt,response pairs) |

Run (example):
```bash
sbatch --array=0-15%8 run_llama_prompts_reddit.slurm
```

## Stage 2: Refusal Classification (`2_Refusal_Classification/`)

Labels each prompt/response pair as refusal or benign using two independent signals, then produces the final training sets.

| File | Purpose |
|------|---------|
| `classify_refusals_local.py` | ML classification via ProtectAI `distilroberta-base-rejection-v1` (M3-optimized, chunked, resumable) |
| `regex_refusals.py` | Regex pattern classifier (lightweight second signal) |
| `classifier_final.py` | Combines both: final_refusals = ML-refusal; final_benign = ML and regex agree NOT refusal |
| `requirements.txt` | Python deps for the classifier |
| `README_refusal_classifier.md` | Detailed docs for the M3 classifier |
| `classified_llama8b_outputs.csv` | ML-only classifications |
| `regexed_outputs.csv` | Regex-only classifications |
| `final_refusals.csv` | All ML-classified refusals (~53k) |
| `final_benign.csv` | Both classifiers agree = benign (~446k) |
| `past_experiments/` | Earlier 5-class classification attempts, disagreement analyses |

Pipeline:
```bash
# 1. ML classification
python classify_refusals_local.py --input llama8b_outputs_filtered.csv --output classified_llama8b_outputs.csv

# 2. Regex classification
python regex_refusals.py --input llama8b_outputs_filtered.csv --output regexed_outputs.csv

# 3. Merge and produce final datasets
python classifier_final.py  # → final_refusals.csv, final_benign.csv
```

## Stage 3: Vector Extraction (`3_Vector_Extraction/`)

Collects activations at the last-k prompt tokens for each class, then computes the difference-of-means vector.

| File | Purpose |
|------|---------|
| `extract_activations_sharded.py` | Shard-aware extractor: collects per-layer activation sums |
| `compute_refusal_vector.py` | Merges shard sums, computes difference-of-means, saves vector |
| `run_extract_activations.slurm` | SLURM array job for activation extraction |
| `run_compute_refusal_vector.slurm` | SLURM job to compute final vector |
| `final_benign_prompts.csv` | Subset of stage-2 benign prompts used for extraction (~40k) |
| `final_refusals_prompts.csv` | Subset of stage-2 refusal prompts used for extraction (~37k) |

Outputs:

| File | Description |
|------|-------------|
| `refusal_vector.npz` | Full vector `[L, H]` across all layers + L2 magnitudes |
| `refusal_vector.layer032.npz` | Single-layer vector (max L2) used by downstream scoring |

Run (<cluster>):
```bash
# 1. Extract benign activations (array job across shards)
sbatch --array=0-159%40 --export=DATASET=benign 3_Vector_Extraction/run_extract_activations.slurm

# 2. Extract refusal activations
sbatch --array=0-159%40 --export=DATASET=refusal 3_Vector_Extraction/run_extract_activations.slurm

# 3. Compute the refusal vector from merged shards
sbatch 3_Vector_Extraction/run_compute_refusal_vector.slurm
```

## Notes

- **Layers**: Aggregates all transformer layers by default. Override with `LAYERS="1,8,16,24"`.
- **Token positions**: Controlled by `LAST_K` (default 1 = final prompt token).
- **Model**: Llama-3-8B-Instruct. Layer 32 has the strongest refusal signal.
- The final vector is consumed by `ppo_or/reward_model.py` (ORRewardModel) and `rwr_overrefusal/score_candidates.py` for computing refusal deltas.
- `refusal_score_demo.ipynb` at this folder's root shows how to load and use the vector for scoring.
