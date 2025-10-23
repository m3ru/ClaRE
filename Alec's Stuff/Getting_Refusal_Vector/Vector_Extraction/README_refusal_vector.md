## Refusal vector (difference-in-means) pipeline

This folder contains a reproducible PACE ICE workflow to extract hidden-state activations from Llama-3 8B Instruct on two datasets (benign and refusal prompts) and compute the refusal direction via difference-in-means.

### Files
- `extract_activations_sharded.py`: shard-aware extractor that aggregates per-layer sums of hidden states over the last-k prompt tokens (no generation). Each shard writes an `.npz` with `sum_by_layer [L,H]` and `count_tokens`.
- `run_extract_activations.slurm`: SLURM array job to run the extractor. Set `DATASET=benign` or `DATASET=refusal`.
- `compute_refusal_vector.py`: merges shard sums for benign and refusal, computes per-layer mean activations and saves `refusal_mean - benign_mean` as `[L,H]` plus L2 magnitudes.
- `run_compute_refusal_vector.slurm`: SLURM job to compute and save the final vectors.

### Prereqs
1) Put your Hugging Face token in the two `.slurm` files: set `HUGGING_FACE_HUB_TOKEN=`.
2) Ensure the prompt CSVs exist on the cluster (default paths below). They can be the large `final_benign_prompts.csv` and `final_refusals_prompts.csv` from this folder copied to `$HOME/scratch/`.

### Default paths on PACE
- Benign prompts: `$HOME/scratch/final_benign_prompts.csv`
- Refusal prompts: `$HOME/scratch/final_refusals_prompts.csv`
- Outputs: shard sums → `$HOME/scratch/acts_benign/*.npz` and `$HOME/scratch/acts_refusal/*.npz`
- Final vector: `$HOME/scratch/refusal_vector.npz`

### Run
1) Submit benign extraction (adjust array size/time as needed):
```bash
sbatch --array=0-159%40 --export=DATASET=benign run_extract_activations.slurm
```
2) Submit refusal extraction:
```bash
sbatch --array=0-159%40 --export=DATASET=refusal run_extract_activations.slurm
```
3) Once all shards finish, compute the refusal vector (defaults to selecting the layer with max L2 magnitude in addition to saving all layers):
```bash
sbatch run_compute_refusal_vector.slurm
```

### Notes
- Layers: by default we aggregate all transformer layers (excluding embeddings). Override with `LAYERS="1,8,16,24"` or `LAYERS=all`.
- Token positions: controlled by `LAST_K` (default 1 = final prompt token). For instruction-style prompts, final post-instruction positions are typically most informative.
- The saved `.npz` at `refusal_vector.npz` contains `vector [L,H]`, `layers [L]`, and `l2_per_layer [L]`. A single-layer vector is also saved to `refusal_vector.layerXXX.npz` when `SELECT_LAYER=max`.


