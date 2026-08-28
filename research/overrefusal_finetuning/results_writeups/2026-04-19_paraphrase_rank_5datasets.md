# Paraphrase + Refusal-Score Ranking — 5 Datasets

**Run date:** 2026-04-19 (scoring finished 2026-04-18 04:10 EDT)
**Scope:** Phase 1 paraphrase (OpenAI Batch, gpt-4o-mini, n=20 paraphrases/prompt) → Phase 2 scoring (Llama-3-8B-Instruct, layer-32 refusal activation) → Phase 3 rank by `mean_or + std_or`, cut top 20%.

## Datasets

| dataset     | n prompts | n paraphrases | top-20% cut |
|-------------|----------:|--------------:|------------:|
| cybermetric |     2,000 |      ~40,000 |         400 |
| sciq        |    11,679 |     ~233,580 |       2,336 |
| medqa       |    10,178 |     ~203,560 |       2,036 |
| pubmedqa    |     1,000 |       20,000 |         200 |
| dolly       |    15,008 |     ~300,160 |       3,002 |

## Top-20% aggregate refusal-shift stats

`mean OR` is the per-prompt mean of paraphrase `or_score` (ranking basis). `mean Δrefusal` is the average of paraphrase_refusal − original_refusal across every paraphrase in the top-20% prompts.

| dataset     | n prompts | mean OR | std OR | mean Δrefusal | std Δ | n paraphrases |
|-------------|----------:|--------:|-------:|--------------:|------:|--------------:|
| cybermetric |       400 |  41.17  | 10.31 |         6.19 | 4.53 |         8,000 |
| sciq        |     2,336 |  43.16  |  9.27 |         6.84 | 4.58 |        46,720 |
| medqa       |     2,036 |  47.39  |  6.23 |         7.06 | 3.43 |        40,720 |
| pubmedqa    |       200 |  57.87  |  9.10 |         7.70 | 4.73 |         4,000 |
| dolly       |     3,002 |  42.87  | 10.43 |         6.70 | 4.84 |        60,029 |

## Absolute refusal scores (top-20%)

Raw layer-32 refusal-vector projections of original vs. paraphrased prompts.

| dataset     | n prompts | mean orig | std orig | mean paraphrase | std paraphrase | Δ (para−orig) |
|-------------|----------:|----------:|---------:|----------------:|---------------:|--------------:|
| cybermetric |       400 |    -4.94 |    1.50 |           1.25 |          4.64 |          6.19 |
| sciq        |     2,336 |    -5.89 |    1.11 |           0.95 |          4.54 |          6.84 |
| medqa       |     2,036 |    -4.06 |    1.29 |           3.00 |          3.36 |          7.06 |
| pubmedqa    |       200 |    -6.31 |    1.00 |           1.40 |          4.71 |          7.70 |
| dolly       |     3,002 |    -5.84 |    1.34 |           0.86 |          4.84 |          6.70 |

## Takeaways

- **Originals don't trigger refusal.** All dataset means are negative (~−4 to −6), confirming the base questions sit well below the refusal-activation threshold.
- **Paraphrasing reliably pushes prompts toward refusal.** Top-20% paraphrases land just above zero in every dataset.
- **pubmedqa** has the strongest per-prompt refusal induction (mean OR 57.9, Δ 7.7) despite having the most negative originals — its top-20% paraphrases cross the zero line by the largest margin relative to baseline.
- **medqa** is the only dataset where top-20% paraphrases average meaningfully positive on their own (+3.0) — those adaptations are refusal-inducing in absolute terms, not just as a delta.
- dolly / sciq / cybermetric cluster together (mean OR 41–43, Δ ~6.2–6.8) — consistent but smaller shifts.

## Artifacts

- Paraphrases: `research/overrefusal_finetuning/paraphrases_api/{dataset}/paraphrases.json`
- Full ranking: `research/overrefusal_finetuning/paraphrases_api/{dataset}/ranked.json`
- Top-20% cut: `research/overrefusal_finetuning/paraphrases_api/{dataset}/top20pct_prompts.json`
- Subset-500 hedge run: `research/overrefusal_finetuning/paraphrases_api/{dataset}/subset500/`

## Compute

- Phase 1: OpenAI Batch API (gpt-4o-mini), 5 parallel batches
- Phase 2+3 (full-scale): <cluster> `<GPU_PARTITION>`, 1× H100, 4h14m
- Phase 2+3 (subset-500 hedge): <cluster> `<GPU_PARTITION>`, 1× L40S, 25m
- Scoring model: `meta-llama/Meta-Llama-3-8B-Instruct`, layer 32, bf16
- Refusal vector: `research/refusal_vector/3_Vector_Extraction/refusal_vector.layer032.npz`
