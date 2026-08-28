## Over-refusal prompt sampling (rejection-sampling style)

This folder generates **(benign_prompt, over_refusal_prompt)** pairs by:
- Taking benign prompts from `databricks-dolly-15k.jsonl`
- Asking multiple frontier models to rewrite each benign prompt into an **over-refusal-triggering** prompt (few-shot, multiple prompt variants)
- Saving many candidates
- Scoring + selecting the “best” candidates for SFT

### Setup

From repo root:

```bash
cd research/overrefusal_sampling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Prepare benign prompt dataset (from Dolly)

```bash
python -m sampling.run_prepare_benign \
  --dolly_jsonl "../overrefusal_finetuning/databricks-dolly-15k.jsonl" \
  --out_csv "data/benign_dolly.csv"
```

By default this **drops** Dolly's `context` field (keeps prompts shorter / cleaner). To include it:

```bash
python -m sampling.run_prepare_benign \
  --dolly_jsonl "../overrefusal_finetuning/databricks-dolly-15k.jsonl" \
  --out_csv "data/benign_dolly.csv" \
  --include_context 1
```

### 2) Sanity check locally (small run)

Paste keys into `sampling/config.py` first.

```bash
python -m sampling.run_generate \
  --benign_csv "data/benign_dolly.csv" \
  --out_jsonl "outputs/samples_sanity.jsonl" \
  --providers "openai,anthropic,gemini,deepseek" \
  --prompt_variants "v1,v2" \
  --limit 5 \
  --num_samples 2 \
  --print_samples 1
```

### 3) Large-scale sampling (<cluster> / Slurm)

See `clusterb/run_sampling_clusterb.slurm`. It shards the benign CSV across an array job and writes JSONL shards to `outputs/`.

### 4) Select best candidates for SFT

```bash
python -m sampling.run_select \
  --in_jsonl_glob "outputs/samples_*.jsonl" \
  --out_csv "outputs/sft_pairs_selected.csv" \
  --top_k_per_prompt 4
```

### Notes

- **Hardcoded keys** live in `sampling/config.py` (you can also override via env vars if you want).
- **Prompt diversity** comes from:
  - multiple providers
  - multiple prompt variants
  - multiple stochastic samples per (provider, variant)
